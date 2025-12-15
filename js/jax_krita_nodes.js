import { app } from "/scripts/app.js"

(function () {
    // Parameter node
    // - adapts to whichever node it is connected to, similar to the built-in "Primitive" node

    const replaceableWidgets = ["INT", "FLOAT", "BOOLEAN", "STRING", "COMBO", "INT:seed"]

    const parameterTypes = {
        "combo": ["choice"],
        "number": ["number", "number (integer)"],
        "toggle": ["toggle"],
        "text": ["text", "prompt (positive)", "prompt (negative)"],
    }

    function defaultParameterType(widgetType, connectedNode, connectedWidget) {
        let paramType = parameterTypes[widgetType][0]
        if (connectedNode.comfyClass === "CLIPTextEncode") {
            paramType = "prompt (positive)"
        }
        if (connectedWidget.options?.round === 1) {
            paramType = "number (integer)"
        }
        return paramType
    }

    function valueMatchesType(value, type, options) {
        if (type === "number") {
            return typeof value === "number"
        } else if (type === "combo") {
            return options?.values?.includes(value)
        } else if (type === "toggle") {
            return typeof value === "boolean"
        }
        return typeof value === "string"
    }

    function optionalWidgetValue(widgets, index, fallback) {
        const result = widgets.length > index ? widgets[index].value : null
        return result === null || result === 0 ? fallback : result
    }

    function changeWidgets(node, type, connectedNode, connectedWidget) {
        if (type === "customtext") {
            type = "text"
        }
        const options = connectedWidget.options

        const parameterTypeHint = node.widgets[1].value
        const notSpecialized = node.widgets[1].options.values.includes("auto")
        const parameterTypeMismatch = !parameterTypes[type].includes(parameterTypeHint)
        if (notSpecialized || parameterTypeMismatch) {
            node.widgets[1].options = { values: parameterTypes[type] }
        }
        if (parameterTypeMismatch) {
            node.widgets[1].value = defaultParameterType(type, connectedNode, connectedWidget)
        }
        const oldDefault = node.widgets.length > 3 ? node.widgets[3].value : connectedWidget.value
        const oldMin = optionalWidgetValue(node.widgets, 4, options?.min ?? 0)
        const oldMax = optionalWidgetValue(node.widgets, 5, options?.max ?? 100)
        const isDefaultValid = valueMatchesType(oldDefault, type, connectedWidget.options)
        while (node.widgets.length > 3) {
            node.widgets.pop()
        }
        const value = isDefaultValid && oldDefault !== "" ? oldDefault : connectedWidget.value
        node.addWidget(type, "default", value, null, options)
        if (type === "number") {
            node.addWidget("number", "min", oldMin, null, options)
            node.addWidget("number", "max", oldMax, null, options)
        }
    }

    function ensureSingleMainSlider(graph) {
        const mainSliderNodes = []
        for (const node of graph._nodes ?? []) {
            if (node?.comfyClass !== "JAX_Parameter") continue
            const mainSliderWidget = node.widgets?.find((w) => w.name === "mainslider")
            if (mainSliderWidget?.value === true) {
                mainSliderNodes.push(node)
            }
        }
        if (mainSliderNodes.length > 1) {
            const names = mainSliderNodes
                .map((n) => n.widgets?.find((w) => w.name === "name")?.value ?? `#${n.id}`)
                .join(", ")
            console.warn(`[jax_krita_nodes] Multiple Parameter nodes have mainslider=true; only the first will be used (${names}).`)
        }
    }

    function enforceNumericMainSlider(node) {
        const typeWidget = node.widgets?.find((w) => w.name === "type")
        const mainSliderWidget = node.widgets?.find((w) => w.name === "mainslider")
        if (!typeWidget || !mainSliderWidget) return

        const isNumeric = typeWidget.value === "number" || typeWidget.value === "number (integer)"
        if (mainSliderWidget.value === true && !isNumeric) {
            mainSliderWidget.value = false
            console.warn(`[jax_krita_nodes] mainslider is only supported for numeric types ("number" or "number (integer)").`)
        }
    }

    function attachMainSliderCallbacks(node) {
        const typeWidget = node.widgets?.find((w) => w.name === "type")
        const mainSliderWidget = node.widgets?.find((w) => w.name === "mainslider")
        if (typeWidget && !typeWidget._jaxMainSliderCallbackAttached) {
            typeWidget._jaxMainSliderCallbackAttached = true
            const old = typeWidget.callback
            typeWidget.callback = function () {
                old?.apply(this, arguments)
                enforceNumericMainSlider(node)
            }
        }
        if (mainSliderWidget && !mainSliderWidget._jaxMainSliderCallbackAttached) {
            mainSliderWidget._jaxMainSliderCallbackAttached = true
            const old = mainSliderWidget.callback
            mainSliderWidget.callback = function () {
                old?.apply(this, arguments)
                enforceNumericMainSlider(node)
                ensureSingleMainSlider(node.graph)
            }
        }
    }

    function adaptWidgetsToConnection(node) {
        if (!node.outputs || node.outputs.length === 0) {
            return
        }
        const links = node.outputs[0].links
        if (links && links.length === 1) {
            const link = node.graph.links[links[0]]
            if (!link) return

            const theirNode = node.graph.getNodeById(link.target_id)
            if (!theirNode || !theirNode.inputs) return

            const input = theirNode.inputs[link.target_slot]
            if (!input || !input.widget || theirNode.widgets === undefined) return

            node.outputs[0].type = input.type

            if (node.widgets[0].value === "Parameter") {
                node.widgets[0].value = input.name
            }

            const widgetName = input.widget.name
            const theirWidget = theirNode.widgets.find((w) => w.name === widgetName)
            if (!theirWidget) return

            const widgetType = theirWidget.origType ?? theirWidget.type
            changeWidgets(node, widgetType, theirNode, theirWidget)
            enforceNumericMainSlider(node)
            attachMainSliderCallbacks(node)
            ensureSingleMainSlider(node.graph)

        } else if (!links || links.length === 0) {
            node.outputs[0].type = "*"
            node.widgets[1].value = "auto"
            node.widgets[1].options = { values: ["auto"] }
            enforceNumericMainSlider(node)
            attachMainSliderCallbacks(node)
            ensureSingleMainSlider(node.graph)
        }
    }

    function setupParameterNode(nodeType) {
        const onAdded = nodeType.prototype.onAdded
        nodeType.prototype.onAdded = function () {
            onAdded?.apply(this, arguments)
            adaptWidgetsToConnection(this)
        }

        const onAfterGraphConfigured = nodeType.prototype.onAfterGraphConfigured
        nodeType.prototype.onAfterGraphConfigured = function () {
            onAfterGraphConfigured?.apply(this, arguments)
            adaptWidgetsToConnection(this)
        }

        const onConnectOutput = nodeType.prototype.onConnectOutput
        nodeType.prototype.onConnectOutput = function (slot, type, input, target_node, target_slot) {
            if (!input.widget && !(input.type in replaceableWidgets)) {
                return false
            } else if (onConnectOutput) {
                const result = onConnectOutput.apply(this, arguments)
                return result
            }
            return true
        }

        const onConnectionsChange = nodeType.prototype.onConnectionsChange
        nodeType.prototype.onConnectionsChange = function (_, index, connected) {
            if (!app.configuringGraph) {
                adaptWidgetsToConnection(this)
            }
            onConnectionsChange?.apply(this, arguments)
        }
    }

    app.registerExtension({
        name: "jax_krita_nodes",
        beforeRegisterNodeDef(nodeType, nodeData) {
            if (nodeData.name === "JAX_Parameter") {
                setupParameterNode(nodeType)
            }
        },
    })
})();
