# Frontend Extensions

Custom nodes can ship frontend JavaScript that adds custom widgets, UI elements, and behaviors to the ComfyUI graph editor. This document covers the key patterns for building frontend extensions, with the eval node as a worked example.

## Extension Registration

Extensions are registered via `app.registerExtension()`:

```javascript
import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "Comfy.MyExtension",

  // Define custom widget types
  getCustomWidgets(app) {
    return {
      MY_WIDGET_TYPE: (node, inputName, inputData) => {
        // Create and return a widget object
      },
    };
  },

  // Modify node definitions before they are registered
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "MyNode") {
      // Patch prototype methods
    }
  },
});
```

## Custom Widgets with DOM Elements

Widgets that need rich DOM (editors, file pickers, media players) follow this pattern:

1. Create the DOM element and append it to `document.body`
2. Position it absolutely over the node using the `draw()` callback
3. Hide it when the node is collapsed, zoomed out, or not in the active graph

### The `draw()` Callback

LiteGraph calls `widget.draw(ctx, node, widgetWidth, y)` each frame for nodes in the **active graph only**. When the user enters a subgraph, `draw()` stops being called for nodes in the parent graph. This means DOM elements positioned by `draw()` will persist at their last position unless explicitly hidden.

### Graph References

There are two graph references that are easy to confuse:

- **`app.graph`**: Always the root graph. Does NOT change when entering subgraphs.
- **`app.canvas.graph`**: The currently displayed graph. Changes when entering/exiting subgraphs.

To check if a node is in the currently visible graph:

```javascript
const isVisible = node.graph === app.canvas.graph;
```

The official frontend (`DomWidgets.vue`) uses the same check:

```javascript
const isInCorrectGraph = posNode.graph === currentGraph;
// where currentGraph = lgCanvas.graph (same as app.canvas.graph)
```

### Hiding DOM Elements on Graph Change

Since `draw()` is not called for nodes outside the active graph, you must use a separate mechanism to hide DOM elements when the user navigates into a subgraph. Two approaches:

**Approach 1: Hook `onDrawForeground`** (runs every frame)

```javascript
const allWidgets = new Set();

const setupHider = () => {
  const canvas = app.canvas;
  const orig = canvas.onDrawForeground;
  canvas.onDrawForeground = function () {
    orig?.apply(this, arguments);
    const currentGraph = app.canvas.graph;
    for (const w of allWidgets) {
      if (w.ownerNode?.graph !== currentGraph) {
        w.domElement.hidden = true;
      }
    }
  };
};
```

**Approach 2: Listen for graph change events**

```javascript
// The canvas dispatches this when setGraph() is called
canvas.canvas.addEventListener("subgraph-opening", (e) => {
  // Hide widgets not in e.detail.subgraph
});
```

## Complete Example: Code Editor Widget

The eval node (`comfy_extras/eval_web/eval_python.js`) demonstrates all these patterns. Here is a simplified version:

```javascript
import { app } from "../../scripts/app.js";

// Track all editor widgets for graph-change hiding
const allEditors = new Set();

// Hide editors whose nodes are not in the active graph.
// This runs every frame via onDrawForeground because draw() is
// not called for nodes outside the active graph.
const setupGraphChangeListener = (() => {
  let installed = false;
  return () => {
    if (installed) return;
    installed = true;
    const tryHook = () => {
      const canvas = app.canvas;
      if (!canvas) { requestAnimationFrame(tryHook); return; }
      const orig = canvas.onDrawForeground;
      canvas.onDrawForeground = function () {
        orig?.apply(this, arguments);
        const currentGraph = app.canvas?.graph;
        if (!currentGraph) return;
        for (const w of allEditors) {
          if (w._ownerNode?.graph !== currentGraph) {
            w.element.hidden = true;
          }
        }
      };
    };
    tryHook();
  };
})();

const createEditorWidget = (node, inputName, inputData) => {
  const widget = {
    type: "my_editor",
    name: inputName,
    options: { hideOnZoom: true },
    value: inputData[1]?.default || "",
    _ownerNode: node,

    // Called every frame ONLY when the node is in the active graph
    draw(ctx, node, widgetWidth, y) {
      const hidden =
        node.flags?.collapsed ||
        (this.options.hideOnZoom && app.canvas.ds.scale < 0.5) ||
        this.type === "converted-widget" ||
        this.type === "hidden";

      this.element.hidden = hidden;
      if (hidden) return;

      // Position the DOM element over the widget area
      // Use ctx.getTransform() to convert canvas coords to screen coords
      const rect = ctx.canvas.getBoundingClientRect();
      const transform = ctx.getTransform();
      const scale = app.canvas.ds.scale;
      const pxRatio = rect.width / ctx.canvas.width;

      Object.assign(this.element.style, {
        position: "absolute",
        left: `${transform.e * pxRatio + rect.left}px`,
        top: `${(transform.f * pxRatio + rect.top) + y * scale}px`,
        width: `${widgetWidth * scale}px`,
        height: `${(node.size[1] - y - 15) * scale}px`,
        zIndex: app.graph._nodes.indexOf(node),
      });
    },

    computeSize() {
      return [500, 250];
    },
  };

  // Create DOM element
  widget.element = document.createElement("div");
  widget.element.hidden = true;
  document.body.appendChild(widget.element);

  // Track for graph-change hiding
  allEditors.add(widget);
  setupGraphChangeListener();

  // Clean up on node removal
  const origRemoved = node.onRemoved;
  node.onRemoved = function () {
    origRemoved?.apply(this, arguments);
    widget.element.remove();
    allEditors.delete(widget);
  };

  return widget;
};

app.registerExtension({
  name: "Comfy.MyEditor",
  getCustomWidgets() {
    return {
      MY_EDITOR: createEditorWidget,
    };
  },
});
```

## Key Pitfalls

1. **`app.graph` vs `app.canvas.graph`**: Always use `app.canvas.graph` for the currently visible graph. `app.graph` is the root and doesn't change on subgraph navigation.

2. **`draw()` only fires for active graph nodes**: DOM elements must be hidden via a separate mechanism (frame callback or event listener) when the user enters a subgraph.

3. **DOM elements on `document.body`**: These persist across graph changes. Always track them and clean up on node removal.

4. **Widget `computeSize()`**: Return `[width, height]` to reserve space in the node layout. The `draw()` callback handles actual positioning of the DOM overlay.

5. **Z-index**: Use `app.graph._nodes.indexOf(node)` or the frontend's `getDomWidgetZIndex()` to layer widgets correctly.

## Eval Nodes

The eval nodes (`EvalPython_1_1`, `EvalPython_5_5`, etc.) allow executing Python code directly in the workflow. They require the `--enable-eval` flag:

```bash
comfyui serve --enable-eval
```

### Variants

| Node | Inputs | Outputs | Description |
|------|--------|---------|-------------|
| EvalPython_1_1 | 1 | 1 | Single value in, single value out |
| EvalPython_5_5 | 5 | 5 | Multiple values in/out |
| EvalPython_1_List | 1 | 1 (list) | Scalar in, list out |
| EvalPython_List_1 | 1 (list) | 1 | List in, scalar out |
| EvalPython_List_List | 1 (list) | 1 (list) | List in, list out |

### Syntax

Use `return` for standard output:

```python
return value0 * 2, value1 + 10
```

Use `yield` for generator output. For scalar nodes, each yield becomes a positional output:

```python
yield value0 * 2
yield value1 + 10
```

For list output nodes, yields are collected into a list:

```python
yield 512
yield 1024
yield 1536
# outputs: [512, 1024, 1536]
```
