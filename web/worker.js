// Runs Pyodide and the AFIS engine off the main thread so the page stays responsive.
// Everything happens inside this browser tab: no image or result is sent anywhere.

const PACKAGES = ["numpy", "scipy", "scikit-image", "opencv-python", "cryptography", "sqlite3"];

let py = null;
let api = null;

function status(text, step) {
  postMessage({ type: "status", text, step });
}

async function boot(version) {
  status("Downloading the Python runtime…", 1);
  importScripts("pyodide/pyodide.js");
  py = await loadPyodide({ indexURL: new URL("pyodide/", self.location).href });

  status("Downloading NumPy, SciPy, scikit-image and OpenCV (about 60 MB, cached after the first visit)…", 2);
  await py.loadPackage(PACKAGES, { messageCallback: () => {}, errorCallback: (m) => console.warn(m) });

  status("Loading the AFIS engine…", 3);
  const zip = await fetch(`afis.zip?v=${encodeURIComponent(version)}`);
  if (!zip.ok) throw new Error(`Could not download the AFIS engine (HTTP ${zip.status})`);
  py.unpackArchive(await zip.arrayBuffer(), "zip", { extractDir: "/home/pyodide/app" });
  py.runPython("import sys; sys.path.insert(0, '/home/pyodide/app')");
  api = py.pyimport("afis_web");

  status("Enrolling the 4 synthetic demo people…", 4);
  const info = api.init().toJs({ dict_converter: Object.fromEntries });
  return info;
}

function call(cmd, args) {
  const fn = api[cmd];
  if (!fn) throw new Error(`Unknown command ${cmd}`);
  const res = fn(...args);
  if (res && typeof res.toJs === "function") {
    const out = res.toJs({ dict_converter: Object.fromEntries });
    res.destroy();
    return out;
  }
  return res;
}

onmessage = async (e) => {
  const { id, cmd, args } = e.data;
  try {
    const result = cmd === "boot" ? await boot(...args) : call(cmd, args);
    postMessage({ type: "result", id, result });
  } catch (err) {
    // Python errors carry the useful message on the last line of the traceback.
    const text = String(err && err.message ? err.message : err).trim().split("\n").pop();
    postMessage({ type: "error", id, error: text });
  }
};
