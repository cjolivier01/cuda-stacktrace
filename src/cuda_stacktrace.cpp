// SPDX-License-Identifier: MIT
// Build: links against CUPTI and pybind11. See setup.py / pyproject.toml.
//
// This extension subscribes to CUPTI callbacks for CUDA Runtime/Driver API
// and, on selected API names, prints the Python call stack at the moment
// of the call.

#include <atomic>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

// Build a CPython extension (no pybind11) to avoid extra deps.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Minimal CUPTI forward declarations to avoid requiring CUDA headers at build
// time.
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

// Known values from CUPTI headers (stable ABI)
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

// Partial struct definition: we only access the first two fields.
typedef struct {
  CUpti_ApiCallbackSite callbackSite;
  const char *functionName;
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
// - Callback API with enter/exit site (CUpti_ApiCallbackSite)
// - For runtime/driver domains, CUpti_CallbackData::functionName is the API
// name.
//   NVIDIA docs state functionName is a global constant and valid to read.
//   [docs] https://docs.nvidia.com/cupti/api/structCUpti__CallbackData.html

// ---------- Helpers & Globals ----------

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

// Watched function names (exact match). Protected by mutex.
static std::unordered_set<std::string> g_filters;
static std::mutex g_filters_mu;

// Domains and site we enabled
static std::atomic<bool> g_domain_runtime{true};
static std::atomic<bool> g_domain_driver{false};
static std::atomic<CUpti_ApiCallbackSite> g_site{CUPTI_API_ENTER};

// Python objects cached (borrowed/owned refs)
static PyObject *g_traceback_mod = nullptr;
static PyObject *g_format_stack_fn = nullptr;

// Small reentrancy guard to avoid recursion if Python printing triggers
// callbacks
thread_local bool tls_in_callback = false;

// ---------- Python stack printing ----------

// Try to join a Python list[str] into a single UTF-8 std::string without
// allocating too much in C++.
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

static void ensure_traceback_objects_held_GIL() {
  if (g_traceback_mod && g_format_stack_fn)
    return;
  g_traceback_mod = PyImport_ImportModule("traceback");
  if (!g_traceback_mod) {
    PyErr_Clear();
    return;
  }
  g_format_stack_fn = PyObject_GetAttrString(g_traceback_mod, "format_stack");
  if (!g_format_stack_fn) {
    PyErr_Clear();
    Py_CLEAR(g_traceback_mod);
  }
}

// Fetch the current Python frame for *this* OS thread, if any.
// We prefer PyThreadState_GetFrame (3.9+) and fall back to PyEval_GetFrame
// (older).
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

static bool should_log_for(const char *functionName) {
  if (!functionName)
    return false;
  std::lock_guard<std::mutex> lock(g_filters_mu);
  if (g_filters.empty())
    return true; // "watch all" if empty
  return (g_filters.find(functionName) != g_filters.end());
}

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

  tls_in_callback = true;

  // Acquire GIL and attempt to format Python stack from this OS thread.
  PyGILState_STATE gil_state = PyGILState_Ensure();

  // Determine whether this thread has Python frames.
  PyObject *frame = get_current_frame_held_GIL(); // new ref (or NULL)
  if (!frame) {
    std::fprintf(
        stderr,
        "[cuda_stacktrace] %s/%s %s — no Python frames in this thread\n",
        domain_str(domain), g_site.load() == CUPTI_API_ENTER ? "enter" : "exit",
        cbInfo->functionName ? cbInfo->functionName : "(unknown)");
    std::fflush(stderr);
    PyGILState_Release(gil_state);
    tls_in_callback = false;
    return;
  }

  ensure_traceback_objects_held_GIL();

  std::string header;
  header.reserve(128);
  header += "[cuda_stacktrace] ";
  header += domain_str(domain);
  header += "/";
  header += (g_site.load() == CUPTI_API_ENTER ? "enter" : "exit");
  header += " ";
  header += (cbInfo->functionName ? cbInfo->functionName : "(unknown)");
  header += "\n";

  if (g_format_stack_fn) {
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

  PyGILState_Release(gil_state);
  tls_in_callback = false;
}

// ---------- Subscription management ----------

// Best-effort: initialize CUDA driver to appease CUPTI on some systems.
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

static PyObject *py_enable(PyObject * /*self*/, PyObject *args,
                           PyObject *kwargs) {
  static const char *kwlist[] = {"api_names", "domains", "site", nullptr};
  PyObject *api_names = nullptr;
  PyObject *domains = nullptr;
  const char *site = "enter";
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Os", (char **)kwlist,
                                   &api_names, &domains, &site)) {
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

  try {
    subscribe_if_needed(runtime, driver);
  } catch (const std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
  g_enabled.store(true);
  Py_RETURN_NONE;
}

static PyObject *py_disable(PyObject * /*self*/, PyObject * /*args*/) {
  g_enabled.store(false);
  unsubscribe_if_needed();
  Py_RETURN_NONE;
}

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
    -1,
    module_methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr};

PyMODINIT_FUNC PyInit__native(void) { return PyModule_Create(&moduledef); }
