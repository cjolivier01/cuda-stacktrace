// SPDX-License-Identifier: MIT
/**
 * @file cuda_stacktrace.cpp
 * @brief CPython extension that prints Python stack traces when selected CUDA
 *        APIs are called via CUPTI callbacks.
 *
 * The module exposes a minimal C API that is wrapped by the Python package
 * `cuda_stacktrace`. It subscribes to CUPTI callbacks for the CUDA Runtime and
 * Driver APIs and, for selected API names, emits the current Python call stack
 * to @c stderr.
 */

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

// Build a CPython extension (no pybind11) to avoid extra deps.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

/// Minimal CUPTI forward declarations to avoid requiring CUDA headers at build time.
#ifndef CUPTIAPI
#ifdef _WIN32
#define CUPTIAPI __stdcall
#else
#define CUPTIAPI
#endif
#endif

typedef int CUptiResult;
typedef void *CUpti_SubscriberHandle;
typedef int CUpti_CallbackDomain;  // enum underlying type
typedef int CUpti_CallbackId;      // enum underlying type
typedef int CUpti_ApiCallbackSite; // enum underlying type

/// Known values from CUPTI headers (stable ABI).
#ifndef CUPTI_SUCCESS
#define CUPTI_SUCCESS 0
#endif
#ifndef CUPTI_CB_DOMAIN_RUNTIME_API
#define CUPTI_CB_DOMAIN_RUNTIME_API 2
#endif
#ifndef CUPTI_CB_DOMAIN_DRIVER_API
#define CUPTI_CB_DOMAIN_DRIVER_API 1
#endif
#ifndef CUPTI_API_ENTER
#define CUPTI_API_ENTER 0
#endif
#ifndef CUPTI_API_EXIT
#define CUPTI_API_EXIT 1
#endif

/// Partial struct definition: we only access the first few fields.
/// The layout matches the CUPTI header for these fields.
typedef struct {
  CUpti_ApiCallbackSite callbackSite;
  const char *functionName;
  const void *functionParams;
  void *functionReturnValue;
} CUpti_CallbackData;

typedef void(CUPTIAPI *CUpti_CallbackFunc)(void *, CUpti_CallbackDomain,
                                           CUpti_CallbackId,
                                           const CUpti_CallbackData *);

extern "C" {
CUptiResult CUPTIAPI cuptiSubscribe(CUpti_SubscriberHandle *subscriber,
                                    CUpti_CallbackFunc cbfunc, void *userdata);
CUptiResult CUPTIAPI cuptiUnsubscribe(CUpti_SubscriberHandle subscriber);
CUptiResult CUPTIAPI cuptiEnableDomain(unsigned int enable,
                                       CUpti_SubscriberHandle subscriber,
                                       CUpti_CallbackDomain domain);
CUptiResult CUPTIAPI cuptiGetResultString(CUptiResult result, const char **str);
}

// Note on what CUPTI provides:
// - Callback API with enter/exit site (CUpti_ApiCallbackSite).
// - For runtime/driver domains, CUpti_CallbackData::functionName is the API name.
//   NVIDIA docs state functionName is a global constant and valid to read.
//   [docs] https://docs.nvidia.com/cupti/api/structCUpti__CallbackData.html

// ---------- Helpers & Globals ----------

/// Helper macro that wraps a CUPTI call and throws std::runtime_error on failure.
#define CUPTI_CALL_THROW(call)                                                 \
  do {                                                                         \
    CUptiResult _status = (call);                                              \
    if (_status != CUPTI_SUCCESS) {                                            \
      const char *_errstr = nullptr;                                           \
      cuptiGetResultString(_status, &_errstr);                                 \
      if (_errstr == nullptr)                                                  \
        _errstr = "Unknown CUPTI error";                                       \
      throw std::runtime_error(std::string("CUPTI error: ") + _errstr);        \
    }                                                                          \
  } while (0)

static std::atomic<bool> g_enabled{false};
static std::atomic<bool> g_subscribed{false};

static CUpti_SubscriberHandle g_subscriber = nullptr;

/// Watched function names (exact match). Protected by mutex.
static std::unordered_set<std::string> g_filters;
static std::mutex g_filters_mu;

/// Domains and site currently configured for callbacks.
static std::atomic<bool> g_domain_runtime{true};
static std::atomic<bool> g_domain_driver{false};
static std::atomic<CUpti_ApiCallbackSite> g_site{CUPTI_API_ENTER};
static std::atomic<bool> g_only_on_error{false};

/// Cached Python traceback module (borrowed/owned references).
static PyObject *g_traceback_mod = nullptr;
static PyObject *g_format_stack_fn = nullptr;
static PyObject *g_extract_stack_fn = nullptr;

/// Small reentrancy guard to avoid recursion if Python printing triggers callbacks.
thread_local bool tls_in_callback = false;

// Optional thread filtering: only log when callback occurs on selected Python
// thread idents. If g_filter_only_current_thread is true, then only the thread
// that called enable() (whose ident is captured) is allowed. If
// g_allowed_thread_idents is non-empty, any thread ident within is allowed.
static std::atomic<bool> g_filter_only_current_thread{false};
static long g_enable_thread_ident = -1; // ident captured at enable()
static std::unordered_set<long> g_allowed_thread_idents; // optional allow-list
static std::mutex g_threads_mu;

// Optional: only print the first stack per originating Python callsite
// (filename + line number of the bottommost Python frame).
static std::atomic<bool> g_once_per_line{false};
static std::unordered_set<std::string> g_seen_callsites;
static std::mutex g_callsites_mu;

// ---------- Python stack printing ----------

/**
 * @brief Join a Python @c list[str] into a single UTF-8 encoded std::string.
 *
 * @param list_obj Python list of strings (borrowed reference).
 * @return UTF-8 encoded concatenation of list elements; empty on error.
 */
static std::string py_list_join_to_string(PyObject *list_obj) {
  std::string out;
  if (!list_obj)
    return out;
  if (!PyList_Check(list_obj))
    return out;

  Py_ssize_t n = PyList_Size(list_obj);
  out.reserve(static_cast<size_t>(n) * 80); // heuristic

  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject *item = PyList_GetItem(list_obj, i); // borrowed
    if (!item)
      continue;
    PyObject *u = PyUnicode_FromObject(item); // PyUnicode or converts
    if (!u)
      continue;
    const char *s = PyUnicode_AsUTF8(u);
    if (s)
      out.append(s);
    Py_DECREF(u);
  }
  return out;
}

/**
 * @brief Ensure the traceback module and helper functions are imported.
 *
 * Must be called with the GIL held. On failure, clears any Python error and
 * leaves the cached pointers null so the caller can handle the absence.
 */
static void ensure_traceback_objects_held_GIL() {
  if (g_traceback_mod && g_format_stack_fn && g_extract_stack_fn)
    return;
  if (!g_traceback_mod) {
    g_traceback_mod = PyImport_ImportModule("traceback");
    if (!g_traceback_mod) {
      PyErr_Clear();
      return;
    }
  }
  if (!g_format_stack_fn) {
    g_format_stack_fn = PyObject_GetAttrString(g_traceback_mod, "format_stack");
    if (!g_format_stack_fn) {
      PyErr_Clear();
      Py_CLEAR(g_traceback_mod);
      return;
    }
  }
  if (!g_extract_stack_fn) {
    g_extract_stack_fn = PyObject_GetAttrString(g_traceback_mod, "extract_stack");
    if (!g_extract_stack_fn) {
      PyErr_Clear();
      // keep going; dedupe feature becomes unavailable
    }
  }
}

/**
 * @brief Fetch the current Python frame for this OS thread (new reference).
 *
 * Uses @c PyThreadState_GetFrame on Python 3.9+ and falls back to
 * @c PyEval_GetFrame on older versions.
 *
 * @return New reference to the current frame object, or @c nullptr if none.
 */
static PyObject *get_current_frame_held_GIL() {
#if PY_VERSION_HEX >= 0x03090000
  PyThreadState *tstate = PyThreadState_Get();
  if (!tstate)
    return nullptr;
  PyObject *frame =
      (PyObject *)PyThreadState_GetFrame(tstate); // new ref or NULL
  return frame;
#else
  PyObject *frame = (PyObject *)PyEval_GetFrame(); // borrowed; may be NULL
  Py_XINCREF(frame);
  return frame;
#endif
}

// ---------- CUPTI callback ----------

/**
 * @brief Check whether a given CUDA API function name should be logged.
 *
 * @param functionName CUDA API name from CUPTI callback data.
 * @return true if logging is enabled for the function; false otherwise.
 */
static bool should_log_for(const char *functionName) {
  if (!functionName)
    return false;
  std::lock_guard<std::mutex> lock(g_filters_mu);
  if (g_filters.empty())
    return true; // "watch all" if empty
  return (g_filters.find(functionName) != g_filters.end());
}

/**
 * @brief Convert a CUPTI callback domain to a human-readable string.
 *
 * @param d CUPTI callback domain enum value.
 * @return "runtime", "driver", or "other".
 */
static const char *domain_str(CUpti_CallbackDomain d) {
  switch (d) {
  case CUPTI_CB_DOMAIN_RUNTIME_API:
    return "runtime";
  case CUPTI_CB_DOMAIN_DRIVER_API:
    return "driver";
  default:
    return "other";
  }
}

/**
 * @brief Build the common header string used for both Python and native stacks.
 */
static std::string build_stack_header(CUpti_CallbackDomain domain,
                                      const CUpti_CallbackData *cbInfo,
                                      bool has_error,
                                      int error_code) {
  std::string header;
  header.reserve(256);
  header += "=================================================================="
            "=======\n";
  header += "[cuda_stacktrace] ";
  header += domain_str(domain);
  header += "/";
  header += (cbInfo && cbInfo->callbackSite == CUPTI_API_ENTER ? "enter"
                                                               : "exit");
  header += " ";
  header += (cbInfo && cbInfo->functionName ? cbInfo->functionName
                                            : "(unknown)");
  if (has_error) {
    header += " error=";
    header += std::to_string(error_code);
  }
  header += "\n";
  header += "------------------------------------------------------------------"
            "-------\n";
  return header;
}

/**
 * @brief Build the footer string used after stack printing.
 */
static std::string build_stack_footer() {
  std::string footer;
  footer.reserve(128);
  footer += "=================================================================="
            "=======\n";
  return footer;
}

/**
 * @brief Best-effort native backtrace to stderr when no Python frames exist.
 *
 * Uses glibc's execinfo routines; demangles symbols when possible and also
 * prints the raw symbol line for addresses.
 */
static void print_native_backtrace() {
#if defined(__linux__) || defined(__APPLE__)
  void *addrs[64];
  int n_frames =
      ::backtrace(addrs, static_cast<int>(sizeof(addrs) / sizeof(addrs[0])));
  if (n_frames <= 0) {
    std::fprintf(stderr, "[cuda_stacktrace] (native backtrace unavailable)\n");
    return;
  }
  char **symbols = ::backtrace_symbols(addrs, n_frames);
  for (int i = 0; i < n_frames; ++i) {
    const char *sym = symbols ? symbols[i] : nullptr;
    if (!sym) {
      std::fprintf(stderr, "  #%02d [unknown]\n", i);
      continue;
    }
    const char *lparen = std::strchr(sym, '(');
    const char *plus = lparen ? std::strchr(lparen, '+') : nullptr;
    std::string demangled;
    if (lparen && plus && lparen + 1 < plus) {
      std::string mangled(lparen + 1, static_cast<size_t>(plus - lparen - 1));
      int status = 0;
      char *dem = abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);
      if (status == 0 && dem) {
        demangled.assign(dem);
      } else {
        demangled.assign(mangled);
      }
      std::free(dem);
    }
    if (!demangled.empty()) {
      std::fprintf(stderr, "  #%02d %s\n", i, demangled.c_str());
      std::fprintf(stderr, "        %s\n", sym);
    } else {
      std::fprintf(stderr, "  #%02d %s\n", i, sym);
    }
  }
  if (symbols)
    std::free(symbols);
#else
  std::fprintf(stderr,
               "[cuda_stacktrace] native backtrace not supported on this platform\n");
#endif
}

/**
 * @brief CUPTI callback invoked for CUDA Runtime/Driver API calls.
 *
 * This function acquires the Python GIL, applies thread and function filters,
 * captures the current Python stack, and prints it to @c stderr. It also
 * implements optional once-per-line deduplication by originating Python
 * callsite.
 *
 * @param userdata User data supplied during subscription (unused).
 * @param domain CUPTI callback domain (runtime or driver).
 * @param cbid CUPTI callback identifier (unused).
 * @param cbInfo Pointer to callback data describing the CUDA API call.
 */
extern "C" void CUPTIAPI cupti_callback(void * /*userdata*/,
                                        CUpti_CallbackDomain domain,
                                        CUpti_CallbackId /*cbid*/,
                                        const CUpti_CallbackData *cbInfo) {

  if (!g_enabled.load(std::memory_order_relaxed))
    return;
  if (!cbInfo)
    return;
  if (tls_in_callback)
    return; // prevent re-entrancy
  if (cbInfo->callbackSite != g_site.load())
    return;
  if (!should_log_for(cbInfo->functionName))
    return;

  int error_code = 0;
  bool has_error = false;
  if (g_only_on_error.load()) {
    if (cbInfo->callbackSite != CUPTI_API_EXIT)
      return;
    if (cbInfo->functionReturnValue) {
      error_code = *reinterpret_cast<const int *>(cbInfo->functionReturnValue);
      has_error = (error_code != 0);
    }
    if (!has_error)
      return;
  }

  tls_in_callback = true;

  // Acquire GIL (needed both for optional thread filtering and stack capture).
  PyGILState_STATE gil_state = PyGILState_Ensure();

  // Optional thread filtering: require the callback to run on one of the
  // allowed Python thread idents.
  if (g_filter_only_current_thread.load() || !g_allowed_thread_idents.empty()) {
    long cur_ident = PyThread_get_thread_ident();
    bool ok = true;
    if (g_filter_only_current_thread.load()) {
      ok = (cur_ident == g_enable_thread_ident);
    }
    if (ok && !g_allowed_thread_idents.empty()) {
      std::lock_guard<std::mutex> lock(g_threads_mu);
      ok = (g_allowed_thread_idents.find(cur_ident) !=
            g_allowed_thread_idents.end());
    }
    if (!ok) {
      PyGILState_Release(gil_state);
      tls_in_callback = false;
      return;
    }
  }

  // Determine whether this thread has Python frames.
  PyObject *frame = get_current_frame_held_GIL(); // new ref (or NULL)
  if (!frame) {
    std::string header = build_stack_header(domain, cbInfo, has_error, error_code);
    std::fwrite(header.data(), 1, header.size(), stderr);
    std::fprintf(stderr,
                 "[cuda_stacktrace] no Python frames in this thread; native "
                 "backtrace:\n");
    print_native_backtrace();
    std::string footer = build_stack_footer();
    std::fprintf(stderr, "%s\n", footer.c_str());
    std::fflush(stderr);
    PyGILState_Release(gil_state);
    tls_in_callback = false;
    return;
  }

  ensure_traceback_objects_held_GIL();

  std::string header = build_stack_header(domain, cbInfo, has_error, error_code);

  if (g_format_stack_fn) {
    // If once-per-line mode is enabled, attempt to extract the bottommost
    // frame's filename+lineno and dedupe on that key.
    if (g_once_per_line.load() && g_extract_stack_fn) {
      PyObject *summ_list = PyObject_CallFunction(g_extract_stack_fn, "O", frame);
      if (!summ_list) {
        PyErr_Clear();
      } else if (PyList_Check(summ_list) && PyList_Size(summ_list) > 0) {
        PyObject *last = PyList_GetItem(summ_list, PyList_Size(summ_list) - 1); // borrowed
        if (last) {
          PyObject *filename = PyObject_GetAttrString(last, "filename");
          PyObject *lineno = PyObject_GetAttrString(last, "lineno");
          const char *fname = nullptr;
          long line = -1;
          if (filename) {
            fname = PyUnicode_Check(filename) ? PyUnicode_AsUTF8(filename) : nullptr;
          }
          if (lineno) {
            if (PyLong_Check(lineno))
              line = PyLong_AsLong(lineno);
          }
          if (fname && line >= 0 && !PyErr_Occurred()) {
            std::string key(fname);
            key.push_back(':');
            key.append(std::to_string(line));
            bool seen = false;
            {
              std::lock_guard<std::mutex> lock(g_callsites_mu);
              auto it = g_seen_callsites.find(key);
              if (it != g_seen_callsites.end()) {
                seen = true;
              } else {
                g_seen_callsites.insert(std::move(key));
              }
            }
            if (seen) {
              // Skip printing entirely; cleanup and exit without emitting header/footer.
              Py_DECREF(frame);
              Py_DECREF(summ_list);
              if (filename)
                Py_DECREF(filename);
              if (lineno)
                Py_DECREF(lineno);
              PyGILState_Release(gil_state);
              tls_in_callback = false;
              return;
            }
          }
          if (filename)
            Py_DECREF(filename);
          if (lineno)
            Py_DECREF(lineno);
        }
        Py_DECREF(summ_list);
      }
    }
    // Call traceback.format_stack(frame)
    PyObject *list_obj = PyObject_CallFunction(g_format_stack_fn, "O", frame);
    Py_DECREF(frame);
    if (!list_obj) {
      PyErr_Clear();
      std::fprintf(stderr, "%s(traceback.format_stack failed)\n",
                   header.c_str());
      std::fflush(stderr);
    } else {
      std::string stack_text = py_list_join_to_string(list_obj);
      Py_DECREF(list_obj);
      std::fwrite(header.data(), 1, header.size(), stderr);
      std::fwrite(stack_text.data(), 1, stack_text.size(), stderr);
      std::fflush(stderr);
    }
  } else {
    Py_DECREF(frame);
    std::fprintf(stderr, "%s(traceback module not available)\n",
                 header.c_str());
    std::fflush(stderr);
  }
  std::string footer = build_stack_footer();
  std::fprintf(stderr, "%s\n", footer.c_str());
  PyGILState_Release(gil_state);
  tls_in_callback = false;
}

// ---------- Subscription management ----------

/**
 * @brief Best-effort attempt to initialize the CUDA driver via @c cuInit().
 *
 * Some systems require the driver to be initialized before CUPTI will accept
 * subscriptions. Failures here are ignored on purpose.
 */
static void try_cuInit() {
  void *h = dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
  if (!h)
    return;
  using cuInit_t = int (*)(unsigned int);
  cuInit_t p_cuInit = (cuInit_t)dlsym(h, "cuInit");
  if (p_cuInit) {
    (void)p_cuInit(0);
  }
  dlclose(h);
}

/**
 * @brief Subscribe to CUPTI callbacks if not already subscribed.
 *
 * Enables the requested runtime/driver domains and installs the global
 * @c cupti_callback handler. If already subscribed, only domain enablement
 * is updated.
 *
 * @param runtime Whether to enable the runtime API domain.
 * @param driver Whether to enable the driver API domain.
 *
 * @throws std::runtime_error if any CUPTI operation fails.
 */
static void subscribe_if_needed(bool runtime, bool driver) {
  if (g_subscribed.load()) {
    // Re-enable/disable domains depending on requested flags.
    CUPTI_CALL_THROW(cuptiEnableDomain(runtime ? 1 : 0, g_subscriber,
                                       CUPTI_CB_DOMAIN_RUNTIME_API));
    CUPTI_CALL_THROW(cuptiEnableDomain(driver ? 1 : 0, g_subscriber,
                                       CUPTI_CB_DOMAIN_DRIVER_API));
    return;
  }
  // Ensure driver is initialized before subscribing (helps avoid
  // CUPTI_ERROR_NOT_INITIALIZED).
  try_cuInit();
  CUPTI_CALL_THROW(cuptiSubscribe(&g_subscriber,
                                  (CUpti_CallbackFunc)cupti_callback, nullptr));
  g_subscribed.store(true);
  CUPTI_CALL_THROW(cuptiEnableDomain(runtime ? 1 : 0, g_subscriber,
                                     CUPTI_CB_DOMAIN_RUNTIME_API));
  CUPTI_CALL_THROW(cuptiEnableDomain(driver ? 1 : 0, g_subscriber,
                                     CUPTI_CB_DOMAIN_DRIVER_API));
}

/**
 * @brief Unsubscribe from CUPTI callbacks if currently subscribed.
 *
 * Attempts to disable all domains and releases the CUPTI subscriber handle.
 * Errors are ignored deliberately to keep teardown best-effort.
 */
static void unsubscribe_if_needed() {
  if (!g_subscribed.load())
    return;
  // Best-effort disable and unsubscribe.
  (void)cuptiEnableDomain(0, g_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
  (void)cuptiEnableDomain(0, g_subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
  (void)cuptiUnsubscribe(g_subscriber);
  g_subscriber = nullptr;
  g_subscribed.store(false);
}

// ---------- CPython Module API ----------

/**
 * @brief Convert a Python iterable of strings into a @c std::vector<std::string>.
 *
 * @param obj Python iterable object (borrowed reference) or @c nullptr.
 * @param out Output vector that will be cleared and filled with UTF-8 strings.
 * @param what Human-readable description used for error messages.
 * @return true on success; false if a Python error is raised.
 */
static bool convert_iterable_of_str(PyObject *obj,
                                    std::vector<std::string> &out,
                                    const char *what) {
  out.clear();
  if (!obj)
    return true; // treat NULL as empty
  PyObject *it = PyObject_GetIter(obj);
  if (!it) {
    PyErr_Format(PyExc_TypeError, "%s must be iterable of str", what);
    return false;
  }
  PyObject *item;
  while ((item = PyIter_Next(it))) {
    PyObject *u = PyUnicode_FromObject(item);
    Py_DECREF(item);
    if (!u) {
      Py_DECREF(it);
      PyErr_Format(PyExc_TypeError, "%s contains non-string", what);
      return false;
    }
    const char *s = PyUnicode_AsUTF8(u);
    if (!s) {
      Py_DECREF(u);
      Py_DECREF(it);
      PyErr_Format(PyExc_TypeError, "%s contains non-UTF8 string", what);
      return false;
    }
    out.emplace_back(s);
    Py_DECREF(u);
  }
  Py_DECREF(it);
  if (PyErr_Occurred())
    return false; // from iteration
  return true;
}

/**
 * @brief Python binding for @c enable().
 *
 * Parses arguments, updates global filters and thread settings, and subscribes
 * to CUPTI as needed.
 *
 * @param self Unused module object.
 * @param args Positional arguments.
 * @param kwargs Keyword arguments.
 * @return @c Py_None on success, or @c nullptr on error.
 */
static PyObject *py_enable(PyObject * /*self*/, PyObject *args,
                           PyObject *kwargs) {
  static const char *kwlist[] = {"api_names",     "domains",
                                 "site",          "only_current_thread",
                                 "thread_idents", "once_per_line",
                                 "only_on_error",
                                 nullptr};
  PyObject *api_names = nullptr;
  PyObject *domains = nullptr;
  const char *site = "enter";
  int only_current_thread = 0;
  PyObject *thread_idents = nullptr;
  int once_per_line = 0;
  int only_on_error = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OspOii", (char **)kwlist,
                                   &api_names, &domains, &site,
                                   &only_current_thread, &thread_idents,
                                   &once_per_line, &only_on_error)) {
    return nullptr;
  }

  std::vector<std::string> names_vec;
  if (!convert_iterable_of_str(api_names, names_vec, "api_names"))
    return nullptr;

  // Update filters
  {
    std::lock_guard<std::mutex> lock(g_filters_mu);
    g_filters.clear();
    for (const auto &s : names_vec)
      g_filters.insert(s);
  }

  // Domains
  bool runtime = false, driver = false;
  if (domains == nullptr || domains == Py_None) {
    runtime = true; // default
  } else {
    std::vector<std::string> doms;
    if (!convert_iterable_of_str(domains, doms, "domains"))
      return nullptr;
    for (const auto &ds : doms) {
      if (ds == "runtime")
        runtime = true;
      else if (ds == "driver")
        driver = true;
      else {
        PyErr_SetString(PyExc_ValueError,
                        "domain must be 'runtime' or 'driver'");
        return nullptr;
      }
    }
    if (!runtime && !driver)
      runtime = true;
  }

  g_domain_runtime.store(runtime);
  g_domain_driver.store(driver);

  // Site
  if (std::strcmp(site, "enter") == 0)
    g_site.store(CUPTI_API_ENTER);
  else if (std::strcmp(site, "exit") == 0)
    g_site.store(CUPTI_API_EXIT);
  else {
    PyErr_SetString(PyExc_ValueError, "site must be 'enter' or 'exit'");
    return nullptr;
  }
  if (only_on_error && g_site.load() != CUPTI_API_EXIT) {
    PyErr_SetString(PyExc_ValueError,
                    "only_on_error requires site='exit'");
    return nullptr;
  }

  // Thread filters
  g_filter_only_current_thread.store(only_current_thread != 0);
  if (g_filter_only_current_thread.load()) {
    // Capture the ident of the Python thread that invoked enable().
    g_enable_thread_ident = PyThread_get_thread_ident();
  } else {
    g_enable_thread_ident = -1;
  }
  {
    std::lock_guard<std::mutex> lock(g_threads_mu);
    g_allowed_thread_idents.clear();
  }
  if (thread_idents && thread_idents != Py_None) {
    PyObject *it = PyObject_GetIter(thread_idents);
    if (!it) {
      PyErr_SetString(PyExc_TypeError,
                      "thread_idents must be an iterable of int");
      return nullptr;
    }
    PyObject *item;
    std::unordered_set<long> tmp;
    while ((item = PyIter_Next(it))) {
      long long v = PyLong_AsLongLong(item);
      Py_DECREF(item);
      if (PyErr_Occurred()) {
        Py_DECREF(it);
        PyErr_SetString(PyExc_TypeError, "thread_idents contains non-integer");
        return nullptr;
      }
      tmp.insert((long)v);
    }
    Py_DECREF(it);
    if (PyErr_Occurred())
      return nullptr;
    if (!tmp.empty()) {
      std::lock_guard<std::mutex> lock(g_threads_mu);
      g_allowed_thread_idents.swap(tmp);
    }
  }

  try {
    subscribe_if_needed(runtime, driver);
  } catch (const std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
  // Configure once-per-line dedupe state
  g_once_per_line.store(once_per_line != 0);
  g_only_on_error.store(only_on_error != 0);
  {
    std::lock_guard<std::mutex> lock(g_callsites_mu);
    g_seen_callsites.clear();
  }
  g_enabled.store(true);
  Py_RETURN_NONE;
}

/**
 * @brief Python binding for @c disable().
 *
 * Disables logging, clears thread filters and deduplication state, and
 * unsubscribes from CUPTI if necessary.
 *
 * @param self Unused module object.
 * @param args Unused positional arguments.
 * @return @c Py_None.
 */
static PyObject *py_disable(PyObject * /*self*/, PyObject * /*args*/) {
  g_enabled.store(false);
  unsubscribe_if_needed();
  // Clear thread filters
  g_filter_only_current_thread.store(false);
  g_enable_thread_ident = -1;
  {
    std::lock_guard<std::mutex> lock(g_threads_mu);
    g_allowed_thread_idents.clear();
  }
  // Clear dedupe state
  g_once_per_line.store(false);
  g_only_on_error.store(false);
  {
    std::lock_guard<std::mutex> lock(g_callsites_mu);
    g_seen_callsites.clear();
  }
  Py_RETURN_NONE;
}

/**
 * @brief Python binding for @c set_functions().
 *
 * Replaces the current function allow-list without modifying the enabled state.
 *
 * @param self Unused module object.
 * @param args Positional arguments containing the iterable of function names.
 * @return @c Py_None on success, or @c nullptr on error.
 */
static PyObject *py_set_functions(PyObject * /*self*/, PyObject *args) {
  PyObject *api_names = nullptr;
  if (!PyArg_ParseTuple(args, "O", &api_names))
    return nullptr;
  std::vector<std::string> names_vec;
  if (!convert_iterable_of_str(api_names, names_vec, "api_names"))
    return nullptr;
  std::lock_guard<std::mutex> lock(g_filters_mu);
  g_filters.clear();
  for (const auto &s : names_vec)
    g_filters.insert(s);
  Py_RETURN_NONE;
}

/**
 * @brief Python binding for @c is_enabled().
 *
 * @param self Unused module object.
 * @param args Unused positional arguments.
 * @return @c Py_True if logging is enabled; @c Py_False otherwise.
 */
static PyObject *py_is_enabled(PyObject * /*self*/, PyObject * /*args*/) {
  if (g_enabled.load())
    Py_RETURN_TRUE;
  Py_RETURN_FALSE;
}

static PyMethodDef module_methods[] = {
    {"enable", (PyCFunction)py_enable, METH_VARARGS | METH_KEYWORDS,
     "Enable stack printing for selected CUDA API names."},
    {"disable", (PyCFunction)py_disable, METH_NOARGS,
     "Disable stack printing and detach CUPTI."},
    {"set_functions", (PyCFunction)py_set_functions, METH_VARARGS,
     "Update the allow-list of CUDA API names."},
    {"is_enabled", (PyCFunction)py_is_enabled, METH_NOARGS,
     "Return whether logging is currently enabled."},
    {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    // Fully qualified module name to live under the Python package
    "cuda_stacktrace._native",
    "Print Python stack whenever selected CUDA APIs are called (via CUPTI).",
    -1, module_methods, nullptr, nullptr, nullptr, nullptr};

/**
 * @brief Module initialization entry point for the CPython extension.
 *
 * @return Newly created module object.
 */
PyMODINIT_FUNC PyInit__native(void) { return PyModule_Create(&moduledef); }
