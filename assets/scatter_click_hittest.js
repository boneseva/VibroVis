/**
 * assets/scatter_click_hittest.js
 * 
 * VibroVis — Marker-Aware Click Selection
 * 
 * Replaces Euclidean center-to-cursor nearest-neighbor selection with
 * screen-space marker-aware hit testing for variable-size markers.
 * 
 * Problem Solved:
 * In Plotly scattergl, large markers (e.g. merged clips with diameter up to 80px)
 * cannot be clicked reliably near their perimeter because Plotly's Euclidean hit-tester
 * picks whichever point center is closest to the cursor. When a small marker (10px)
 * is nearby, clicking inside the large marker's body selects the small marker instead.
 * 
 * Solution:
 * 1. Maintain a fast 2D spatial index of rendered points with their screen-space
 *    marker bounding circles (radius = marker_size / 2).
 * 2. On click, find all markers that physically contain the click position (d <= radius).
 * 3. Deterministic tie-breaking:
 *    - Primary: Trace draw order (higher curveNumber rendered on top).
 *    - Secondary: Point draw order within trace (higher pointNumber rendered on top).
 * 4. Boundary tolerance fallback: if click is just outside a marker edge (within 5px),
 *    select the closest marker perimeter (min d - radius).
 * 5. Dispatches authoritative clickData payload to Dash callbacks:
 *    - gd.emit('plotly_click', payload)
 *    - window.dash_clientside.set_props('scatter', { clickData: payload })
 *    - Stops native event propagation when a marker is hit to prevent Plotly from
 *      overwriting with the wrong point.
 * 6. Zero DOM overlays, zero stems/bubbles/tooltips, zero hover interference, zero flickering.
 */
(function() {
    'use strict';

    if (window._vibrovisHitTestInitialized === 'v1') {
        return;
    }
    window._vibrovisHitTestInitialized = 'v1';

    // Constants
    const GRID_BIN_SIZE = 40;              // 2D Spatial uniform grid bin size in px
    const MAX_MARKER_RADIUS = 40;          // Max marker radius (diameter 80px / 2)
    const BOUNDARY_TOLERANCE = 5;          // Edge proximity tolerance in px for near-miss clicks
    const CLICK_DEBOUNCE_MS = 200;         // Debounce guard to prevent duplicate dispatches

    // State
    let scatterEl = null;
    let spatialGrid = null;
    let isDispatchingClick = false;

    /**
     * 2D Spatial uniform grid for sub-millisecond point range queries.
     */
    class MarkerSpatialGrid {
        constructor(binSize = 40) {
            this.binSize = binSize;
            this.bins = new Map();
            this.totalPoints = 0;
        }

        _key(bx, by) {
            return `${bx},${by}`;
        }

        insert(pt) {
            const bx = Math.floor(pt.screenX / this.binSize);
            const by = Math.floor(pt.screenY / this.binSize);
            const key = this._key(bx, by);
            let bin = this.bins.get(key);
            if (!bin) {
                bin = [];
                this.bins.set(key, bin);
            }
            bin.push(pt);
            this.totalPoints++;
        }

        /**
         * Query all points whose centers are within searchRadius of (x, y).
         */
        queryRadius(x, y, searchRadius) {
            const results = [];
            const r2 = searchRadius * searchRadius;
            const minBx = Math.floor((x - searchRadius) / this.binSize);
            const maxBx = Math.floor((x + searchRadius) / this.binSize);
            const minBy = Math.floor((y - searchRadius) / this.binSize);
            const maxBy = Math.floor((y + searchRadius) / this.binSize);

            for (let bx = minBx; bx <= maxBx; bx++) {
                for (let by = minBy; by <= maxBy; by++) {
                    const bin = this.bins.get(this._key(bx, by));
                    if (!bin) continue;
                    for (let i = 0; i < bin.length; i++) {
                        const pt = bin[i];
                        const dx = pt.screenX - x;
                        const dy = pt.screenY - y;
                        const dist2 = dx * dx + dy * dy;
                        if (dist2 <= r2) {
                            results.push({ pt, dist: Math.sqrt(dist2) });
                        }
                    }
                }
            }
            return results;
        }
    }

    /**
     * Extract coordinate array handling raw arrays, TypedArrays,
     * Dash binary buffer objects ({ dtype, bdata, _inputArray }), and base64.
     */
    function getCoordinateArray(val) {
        if (!val) return [];
        if (Array.isArray(val) || ArrayBuffer.isView(val)) return val;
        if (val._inputArray && (Array.isArray(val._inputArray) || ArrayBuffer.isView(val._inputArray))) {
            return val._inputArray;
        }
        if (val.bdata && typeof atob === 'function') {
            try {
                const binStr = atob(val.bdata);
                const len = binStr.length;
                const bytes = new Uint8Array(len);
                for (let i = 0; i < len; i++) bytes[i] = binStr.charCodeAt(i);
                if (val.dtype === 'f4' || val.dtype === '<f4') return new Float32Array(bytes.buffer);
                if (val.dtype === 'i4' || val.dtype === '<i4') return new Int32Array(bytes.buffer);
                if (val.dtype === 'f8' || val.dtype === '<f8') return new Float64Array(bytes.buffer);
                return bytes;
            } catch (e) {
                return [];
            }
        }
        return [];
    }

    /**
     * Build spatial index from the Plotly graph DOM element.
     */
    function buildSpatialIndex(gd) {
        if (!gd || !gd._fullLayout || !gd._fullLayout.xaxis || !gd._fullLayout.yaxis || !gd.data) {
            return null;
        }

        const fullLayout = gd._fullLayout;
        const xaxis = fullLayout.xaxis;
        const yaxis = fullLayout.yaxis;
        const leftMargin = (typeof xaxis._offset === 'number') ? xaxis._offset : (fullLayout._size ? fullLayout._size.l : (fullLayout.margin ? fullLayout.margin.l : 0));
        const topMargin = (typeof yaxis._offset === 'number') ? yaxis._offset : (fullLayout._size ? fullLayout._size.t : (fullLayout.margin ? fullLayout.margin.t : 0));

        const grid = new MarkerSpatialGrid(GRID_BIN_SIZE);

        for (let curveIdx = 0; curveIdx < gd.data.length; curveIdx++) {
            const trace = gd.data[curveIdx];
            if (!trace || !trace.x || !trace.y || trace.visible === false || trace.visible === 'legendonly') {
                continue;
            }

            const xArr = getCoordinateArray(trace.x);
            const yArr = getCoordinateArray(trace.y);
            const cdArr = trace.customdata;
            const sizeArr = (trace.marker && trace.marker.size) ? getCoordinateArray(trace.marker.size) : null;
            const isScalarSize = (trace.marker && typeof trace.marker.size === 'number') ? trace.marker.size : null;

            const len = Math.min(xArr.length, yArr.length);
            for (let ptIdx = 0; ptIdx < len; ptIdx++) {
                const xVal = xArr[ptIdx];
                const yVal = yArr[ptIdx];

                // Skip non-finite values
                if (typeof xVal !== 'number' || typeof yVal !== 'number' || isNaN(xVal) || isNaN(yVal)) {
                    continue;
                }

                // Plotly c2p coordinate transform
                const px = leftMargin + xaxis.c2p(xVal);
                const py = topMargin + yaxis.c2p(yVal);

                if (isNaN(px) || isNaN(py)) continue;

                const cd = (cdArr && cdArr[ptIdx]) ? cdArr[ptIdx] : null;

                // Determine marker diameter in pixels:
                // 1. From size array / buffer if available
                // 2. From scalar trace.marker.size if present
                // 3. From customdata[4] (clip_count) using formula: 10 + 3 * (clip_count - 1), clamped to [10, 80]
                let diameter = 10;
                if (sizeArr && typeof sizeArr[ptIdx] === 'number') {
                    diameter = sizeArr[ptIdx];
                } else if (isScalarSize !== null) {
                    diameter = isScalarSize;
                } else if (cd && typeof cd[4] === 'number') {
                    diameter = 10 + 3 * (cd[4] - 1);
                }
                diameter = Math.min(80, Math.max(10, diameter));
                const radius = diameter / 2;

                grid.insert({
                    curveNumber: curveIdx,
                    pointNumber: ptIdx,
                    dataX: xVal,
                    dataY: yVal,
                    screenX: px,
                    screenY: py,
                    diameter: diameter,
                    radius: radius,
                    customdata: cd,
                    traceName: trace.name || `Trace ${curveIdx}`
                });
            }
        }

        return grid;
    }

    /**
     * Resolve which marker was clicked using screen-space marker geometry.
     * 
     * Rules:
     * 1. Candidates whose visible circle physically contains the cursor (dist <= radius)
     *    are prioritized over points outside.
     * 2. Overlapping candidates are tie-broken deterministically by rendering order:
     *    - curveNumber descending (later traces rendered on top)
     *    - pointNumber descending (later points within trace rendered on top)
     * 3. If no marker contains the cursor, candidates within BOUNDARY_TOLERANCE (5px)
     *    of a marker's outer edge are evaluated by minimum edge distance (dist - radius).
     * 4. If nothing is within tolerance, returns null (empty background click).
     */
    function resolveMarkerHit(mx, my) {
        if (!spatialGrid || spatialGrid.totalPoints === 0) return null;

        const maxQueryRadius = MAX_MARKER_RADIUS + BOUNDARY_TOLERANCE;
        const candidates = spatialGrid.queryRadius(mx, my, maxQueryRadius);
        if (candidates.length === 0) return null;

        // Step 1: Collect candidates whose marker body physically contains the cursor
        const insideCandidates = [];
        for (let i = 0; i < candidates.length; i++) {
            const { pt, dist } = candidates[i];
            if (dist <= pt.radius) {
                insideCandidates.push({ pt, dist });
            }
        }

        if (insideCandidates.length > 0) {
            // Tie-break by rendering order (topmost trace, topmost point within trace)
            insideCandidates.sort((a, b) => {
                if (b.pt.curveNumber !== a.pt.curveNumber) {
                    return b.pt.curveNumber - a.pt.curveNumber;
                }
                return b.pt.pointNumber - a.pt.pointNumber;
            });
            return insideCandidates[0].pt;
        }

        // Step 2: Near-miss boundary tolerance (click just outside marker edge)
        let edgeCandidates = [];
        for (let i = 0; i < candidates.length; i++) {
            const { pt, dist } = candidates[i];
            const edgeDist = dist - pt.radius;
            if (edgeDist <= BOUNDARY_TOLERANCE) {
                edgeCandidates.push({ pt, edgeDist });
            }
        }

        if (edgeCandidates.length > 0) {
            edgeCandidates.sort((a, b) => {
                const diff = a.edgeDist - b.edgeDist;
                if (Math.abs(diff) > 0.5) return diff;
                if (b.pt.curveNumber !== a.pt.curveNumber) {
                    return b.pt.curveNumber - a.pt.curveNumber;
                }
                return b.pt.pointNumber - a.pt.pointNumber;
            });
            return edgeCandidates[0].pt;
        }

        return null;
    }

    /**
     * Dispatch authoritative Dash clickData payload to all downstream callbacks.
     */
    function dispatchPointSelection(gd, pt) {
        const clickPayload = {
            points: [{
                curveNumber: pt.curveNumber,
                pointNumber: pt.pointNumber,
                pointIndex: pt.pointNumber,
                x: pt.dataX,
                y: pt.dataY,
                customdata: pt.customdata,
                data: (gd.data && gd.data[pt.curveNumber]) ? gd.data[pt.curveNumber] : undefined
            }]
        };

        // 1. Emit through Plotly event emitter
        if (typeof gd.emit === 'function') {
            gd.emit('plotly_click', clickPayload);
        }

        // 2. Dispatch directly to Dash Component API
        if (window.dash_clientside && typeof window.dash_clientside.set_props === 'function') {
            try {
                window.dash_clientside.set_props('scatter', { clickData: clickPayload });
            } catch (err) {
                console.warn('[VibroVisHitTest] dash_clientside.set_props warning:', err);
            }
        }
    }

    /**
     * Capture-phase click listener on graph container.
     */
    function handlePlotClick(e) {
        if (!spatialGrid) return;
        const gd = scatterEl;
        if (!gd || !gd._fullLayout || !gd._fullLayout.xaxis || !gd._fullLayout.yaxis) return;

        const fullLayout = gd._fullLayout;
        const xaxis = fullLayout.xaxis;
        const yaxis = fullLayout.yaxis;
        const leftMargin = (typeof xaxis._offset === 'number') ? xaxis._offset : 0;
        const topMargin = (typeof yaxis._offset === 'number') ? yaxis._offset : 0;
        const plotWidth = (typeof xaxis._length === 'number') ? xaxis._length : (fullLayout.width || 0);
        const plotHeight = (typeof yaxis._length === 'number') ? yaxis._length : (fullLayout.height || 0);

        const rect = gd.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        // Check if click is inside the active plot area
        if (mx < leftMargin || mx > leftMargin + plotWidth || my < topMargin || my > topMargin + plotHeight) {
            return; // Ignore clicks outside plot area (axes, margins, modebar)
        }

        const winner = resolveMarkerHit(mx, my);
        if (!winner) {
            // No marker hit within tolerance; allow default background click
            return;
        }

        // Marker hit found: prevent Plotly's Euclidean-nearest search from firing
        e.stopImmediatePropagation();
        e.preventDefault();

        if (isDispatchingClick) return;
        isDispatchingClick = true;
        setTimeout(() => { isDispatchingClick = false; }, CLICK_DEBOUNCE_MS);

        dispatchPointSelection(gd, winner);
    }

    /**
     * Find the actual Plotly graph DOM element.
     */
    function findPlotlyElement() {
        const parent = document.getElementById('scatter');
        if (!parent) return null;
        if (parent._fullLayout && parent.data) return parent;
        const child = parent.querySelector('.js-plotly-plot');
        if (child && child._fullLayout && child.data) return child;
        return null;
    }

    /**
     * Bind listeners to graph element and container.
     */
    function bindGraphListeners(gd) {
        if (!gd || gd._vHitTestBound) return;
        gd._vHitTestBound = true;
        scatterEl = gd;

        // Build initial spatial index
        spatialGrid = buildSpatialIndex(gd);

        // Rebuild spatial index whenever plot renders, scales, zooms, or filters change
        if (typeof gd.on === 'function') {
            gd.on('plotly_afterplot', function() {
                spatialGrid = buildSpatialIndex(gd);
            });

            gd.on('plotly_relayout', function() {
                spatialGrid = buildSpatialIndex(gd);
            });

            gd.on('plotly_restyle', function() {
                spatialGrid = buildSpatialIndex(gd);
            });
        }

        // Attach click listener in the CAPTURE phase to intercept clicks before Plotly canvas
        gd.addEventListener('click', handlePlotClick, { capture: true });
        const parentScatter = document.getElementById('scatter');
        if (parentScatter && parentScatter !== gd) {
            parentScatter.addEventListener('click', handlePlotClick, { capture: true });
        }
    }

    /**
     * Compute total number of renderable points in Plotly data array.
     */
    function getGraphPointCount(gd) {
        if (!gd || !gd.data) return 0;
        let count = 0;
        for (let i = 0; i < gd.data.length; i++) {
            const trace = gd.data[i];
            if (trace && trace.x && trace.visible !== false && trace.visible !== 'legendonly') {
                const xArr = getCoordinateArray(trace.x);
                count += xArr.length;
            }
        }
        return count;
    }

    /**
     * Public introspection and test API.
     */
    window.VibroVisHitTest = {
        resolveAt: function(screenX, screenY) {
            if (!spatialGrid) return null;
            return resolveMarkerHit(screenX, screenY);
        },
        clickAt: function(screenX, screenY) {
            const pt = this.resolveAt(screenX, screenY);
            if (pt && scatterEl) {
                dispatchPointSelection(scatterEl, pt);
                return pt;
            }
            return null;
        },
        getStatus: function() {
            return {
                bound: !!scatterEl,
                totalIndexedPoints: spatialGrid ? spatialGrid.totalPoints : 0
            };
        },
        reindex: function() {
            if (scatterEl) {
                spatialGrid = buildSpatialIndex(scatterEl);
            }
            return spatialGrid ? spatialGrid.totalPoints : 0;
        }
    };

    /**
     * Initialization watcher: waits for Plotly graph to exist in the DOM.
     */
    function waitForScatterGraph() {
        const gd = findPlotlyElement();
        if (gd) {
            bindGraphListeners(gd);
        }

        const checkInterval = setInterval(function() {
            const el = findPlotlyElement();
            if (el) {
                if (!scatterEl || el !== scatterEl || !el._vHitTestBound) {
                    bindGraphListeners(el);
                } else {
                    const currentPoints = getGraphPointCount(el);
                    const indexed = spatialGrid ? spatialGrid.totalPoints : 0;
                    if (currentPoints > 0 && indexed !== currentPoints) {
                        spatialGrid = buildSpatialIndex(el);
                    }
                }
            }
        }, 250);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', waitForScatterGraph);
    } else {
        waitForScatterGraph();
    }
})();
