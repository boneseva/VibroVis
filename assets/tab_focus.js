/**
 * tab_focus.js
 * Detects when the user tabs keyboard focus into a table-inline-label dcc.Input,
 * extracts the plot_id index from its Dash pattern-match id, and writes
 * "<index>_<timestamp>" to the hidden focused-row-trigger-input so that a
 * Dash clientside callback can forward it to focused-row-store.
 */
(function () {
    'use strict';

    var _nativeSetter = Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype, 'value'
    ).set;

    document.addEventListener('focusin', function (e) {
        var el = e.target;
        if (!el || el.tagName !== 'INPUT') return;

        var idStr = el.getAttribute('id') || '';
        if (!idStr) return;

        var idObj;
        try {
            idObj = JSON.parse(idStr);
        } catch (_) {
            return;
        }

        if (!idObj || idObj.type !== 'table-inline-label' || idObj.index === undefined) return;

        var trigger = document.getElementById('focused-row-trigger-input');
        if (!trigger) return;

        var newVal = String(idObj.index) + '_' + Date.now();
        _nativeSetter.call(trigger, newVal);
        trigger.dispatchEvent(new Event('input', { bubbles: true }));
    });
}());
