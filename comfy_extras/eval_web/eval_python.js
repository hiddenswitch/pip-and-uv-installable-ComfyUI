/**
 * Uses code adapted from https://github.com/yorkane/ComfyUI-KYNode
 *
 * MIT License
 *
 * Copyright (c) 2024 Kevin Yuan
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
import { app } from "../../scripts/app.js";

// All ace/LSP assets are served from the extension's web directory
const _extBase = "/extensions/comfy_extras.nodes.nodes_eval";

const _loadScript = (path) => new Promise((resolve) => {
  const s = document.createElement("script");
  s.src = `${_extBase}/${path}`;
  s.onload = resolve;
  s.onerror = resolve;
  document.head.appendChild(s);
});

// Load Ace editor and extensions
let ace;
if (window.ace) {
  ace = window.ace;
} else {
  await _loadScript("ace.js");
  ace = window.ace;
}
// Set basePath so ace can find modes/themes/extensions from our directory
ace.config.set("basePath", _extBase);

// Load language_tools for autocompletion (must be after ace)
await _loadScript("ext-language_tools.js");

// Load ace-linters LSP support (optional, degrades gracefully)
// language-client.js provides the LanguageClient class for socket mode
// ace-language-client.js provides AceLanguageClient which wires it to ACE
let AceLanguageClient;
await _loadScript("language-client.js");
await _loadScript("ace-language-client.js");
AceLanguageClient = window.AceLanguageClient;

// LSP provider singleton: connects all editors to /ws/lsp via basedpyright
let lspProvider = null;
let lspEditorCounter = 0;
const getLspProvider = () => {
  if (lspProvider) return lspProvider;
  if (!AceLanguageClient) return null;
  try {
    const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${wsProto}//${window.location.host}/ws/lsp`;
    // language-client.js is already loaded (exports LanguageClient to window)
    // The module callback must return an object with a LanguageClient property
    lspProvider = AceLanguageClient.for({
      modes: "python",
      type: "socket",
      socket: new WebSocket(wsUrl),
      module: () => Promise.resolve(window),
    });
  } catch (e) {
    console.warn("LSP provider failed:", e);
  }
  return lspProvider;
};

const findWidget = (node, value, attr = "name", func = "find") => {
  return node?.widgets ? node.widgets[func]((w) => (Array.isArray(value) ? value.includes(w[attr]) : w[attr] === value)) : null;
};

// Trigger workflow change tracking
const markWorkflowChanged = () => {
  app?.extensionManager?.workflow?.activeWorkflow?.changeTracker?.checkState();
};

// Detect nodes 2.0: addDOMWidget exists and creates managed widgets
const useNodes2 = typeof app?.canvas?.graph?.getNodeById === "function"
  && typeof document.querySelector?.(".dom-widget") !== "undefined";

// Shared: create ACE editor on a container element
const initAceEditor = (container, defaultValue) => {
  const editor = ace.edit(container);
  editor.setTheme("ace/theme/monokai");
  editor.session.setMode("ace/mode/python");
  editor.setOptions({
    enableAutoIndent: true,
    enableLiveAutocompletion: true,
    enableBasicAutocompletion: true,
    enableSnippets: true,
    fontFamily: "monospace",
  });
  // Connect to LSP for code intelligence (autocomplete, imports)
  const provider = getLspProvider();
  if (provider) {
    provider.registerEditor(editor);
    provider.setSessionFilePath(editor.session, {
      filePath: `eval_node_${lspEditorCounter++}.py`,
    });
  }

  if (defaultValue) {
    editor.setValue(defaultValue);
    editor.clearSelection();
  }
  return editor;
};


// ============================================================
// Nodes 2.0: Use addDOMWidget for Vue-managed positioning
// DomWidgets.vue handles subgraph visibility automatically.
// ============================================================

const codeEditorV2 = (node, inputName, inputData) => {
  const defaultValue = inputData[1]?.default || "";

  // Create the container element for the ACE editor
  const container = document.createElement("div");
  container.style.cssText = `
    width: 100%;
    height: 100%;
    min-height: 200px;
    --comfy-widget-min-height: 200;
    --comfy-widget-height: 50%;
  `;

  const editor = initAceEditor(container, defaultValue);

  // Use addDOMWidget: the frontend's DomWidgets.vue manages positioning,
  // visibility, z-index, and subgraph awareness automatically.
  const widget = node.addDOMWidget(inputName, "code_block_python", container, {
    hideOnZoom: true,
    getValue: () => editor.getValue(),
    setValue: (v) => {
      if (editor.getValue() !== v) {
        editor.setValue(v);
        editor.clearSelection();
      }
    },
    getMinHeight: () => 200,
    getHeight: () => "50%",
  });

  // Store editor reference for onConfigure
  widget.editor = editor;

  editor.getSession().on("change", () => {
    markWorkflowChanged();
  });

  // Resize editor when widget resizes
  widget.options.afterResize = () => {
    editor.resize();
  };

  return widget;
};


// ============================================================
// Nodes 1.0 (legacy): Manual DOM positioning with draw()
// Requires explicit subgraph-change hiding via onDrawForeground.
// ============================================================

const allCodeWidgetsV1 = new Set();

const setupGraphChangeListenerV1 = (() => {
  let installed = false;
  return () => {
    if (installed) return;
    installed = true;

    const hideStaleEditors = () => {
      // app.canvas.graph is the currently displayed graph (follows subgraph
      // navigation). app.graph stays on the root graph and must not be used.
      const currentGraph = app.canvas?.graph;
      if (!currentGraph) return;
      for (const w of allCodeWidgetsV1) {
        if (w._ownerNode?.graph !== currentGraph) {
          w.codeElement.hidden = true;
        }
      }
    };

    const tryHook = () => {
      const canvas = app.canvas;
      if (!canvas) {
        requestAnimationFrame(tryHook);
        return;
      }
      const origDraw = canvas.onDrawForeground;
      canvas.onDrawForeground = function () {
        origDraw?.apply(this, arguments);
        hideStaleEditors();
      };
    };
    tryHook();
  };
})();

const getPositionV1 = (node, ctx, w_width, y, n_height) => {
  const margin = 5;
  const rect = ctx.canvas.getBoundingClientRect();
  const transform = ctx.getTransform();
  const scale = app.canvas.ds.scale;
  const canvasPixelToScreenPixel = rect.width / ctx.canvas.width;
  const x = transform.e * canvasPixelToScreenPixel + rect.left;
  const y_pos = transform.f * canvasPixelToScreenPixel + rect.top;
  const scaledWidth = w_width * scale;
  const scaledHeight = (n_height - y - 15) * scale;
  const scaledMargin = margin * scale;
  const scaledY = y * scale;

  return {
    left: `${x + scaledMargin}px`,
    top: `${y_pos + scaledY + scaledMargin}px`,
    width: `${scaledWidth - scaledMargin * 2}px`,
    maxWidth: `${scaledWidth - scaledMargin * 2}px`,
    height: `${scaledHeight - scaledMargin * 2}px`,
    maxHeight: `${scaledHeight - scaledMargin * 2}px`,
    position: "absolute",
    scrollbarColor: "var(--descrip-text) var(--bg-color)",
    scrollbarWidth: "thin",
    zIndex: app.graph._nodes.indexOf(node),
  };
};

const codeEditorV1 = (node, inputName, inputData) => {
  const widget = {
    type: "code_block_python",
    name: inputName,
    options: { hideOnZoom: true },
    value: inputData[1]?.default || "",
    _ownerNode: node,
    draw(ctx, node, widgetWidth, y) {
      const hidden = node.flags?.collapsed || (!!this.options.hideOnZoom && app.canvas.ds.scale < 0.5) || this.type === "converted-widget" || this.type === "hidden";

      this.codeElement.hidden = hidden;

      if (hidden) {
        this.options.onHide?.(this);
        return;
      }

      Object.assign(this.codeElement.style, getPositionV1(node, ctx, widgetWidth, y, node.size[1]));
    },
    computeSize() {
      return [500, 250];
    },
  };

  widget.codeElement = document.createElement("pre");
  widget.codeElement.textContent = widget.value;

  widget.editor = initAceEditor(widget.codeElement, widget.value);
  widget.codeElement.hidden = true;

  document.body.appendChild(widget.codeElement);
  allCodeWidgetsV1.add(widget);

  const originalCollapse = node.collapse;
  node.collapse = function () {
    originalCollapse.apply(this, arguments);
    widget.codeElement.hidden = !!this.flags?.collapsed;
  };

  setupGraphChangeListenerV1();

  return widget;
};


// ============================================================
// Register extension: picks v2 or v1 based on addDOMWidget
// ============================================================

app.registerExtension({
  name: "Comfy.EvalPython",
  getCustomWidgets(app) {
    return {
      CODE_BLOCK_PYTHON: (node, inputName, inputData) => {
        // Nodes 2.0: use addDOMWidget for Vue-managed lifecycle
        if (typeof node.addDOMWidget === "function") {
          const widget = codeEditorV2(node, inputName, inputData);

          widget.editor.getSession().on("change", () => {
            // widget.value is managed by getValue/setValue options
            markWorkflowChanged();
          });

          return widget;
        }

        // Nodes 1.0 fallback: manual DOM management
        const widget = codeEditorV1(node, inputName, inputData);

        widget.editor.getSession().on("change", () => {
          widget.value = widget.editor.getValue();
          markWorkflowChanged();
        });

        node.onRemoved = function () {
          for (const w of this.widgets) {
            if (w?.codeElement) {
              w.codeElement.remove();
              allCodeWidgetsV1.delete(w);
            }
          }
        };

        node.addCustomWidget(widget);

        return widget;
      },
    };
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    // Handle all EvalPython node variants
    if (nodeData.name.startsWith("EvalPython")) {
      const originalOnConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        originalOnConfigure?.apply(this, arguments);

        if (info?.widgets_values?.length) {
          const widgetCodeIndex = findWidget(this, "code_block_python", "type", "findIndex");
          const editor = this.widgets[widgetCodeIndex]?.editor;

          if (editor) {
            editor.setValue(info.widgets_values[widgetCodeIndex]);
            editor.clearSelection();
          }
        }
      };
    }
  },
});
