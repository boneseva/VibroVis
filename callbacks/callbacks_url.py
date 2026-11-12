"""
URL initialisation and restoration callbacks: the server side of "is there a view to restore, and which parts of it
are valid for this browser's dataset?".

Nothing here writes a widget. ``url_init`` parses the URL; ``url_plan_filters`` validates the filter values against the
browser's OWN loaded dataset. The browser-side reconciler (assets/url_reconcile.js) applies the result with set_props.

Data flow (single writer for every Store):

  assets/url_restore.js  arbiter (client side)  --writes-->  url-init-request {epoch, search}      (starts an attempt)
  url_init (below)       reads the request      --writes-->  url-init-response {epoch, payload}    (an EVENT, not state)
  arbiter                reads the response as STATE        url-init  (the PUBLIC state: pending/running/ready/failed)
  load_data_into_server_cache   reads url-init and holds while pending/running.

The server's answer carries the epoch of the attempt it belongs to; the arbiter ignores answers for any other epoch and any
answer that arrives after the attempt ended (see assets/url_restore.js).
"""
from dash import ClientsideFunction, Input, Output, State
from dash.exceptions import PreventUpdate

import limits
import url_plan
import url_state
from callbacks import dataset_access

MAX_SEARCH_CHARS = 100_000          # never parse an absurdly long query; url_state reports query_too_long above its limit


def register_url_callbacks(app):

    @app.callback(
        Output('url-init-response', 'data'),
        Input('url-init-request', 'data'),
        prevent_initial_call=True,
        # `running` is how the browser learns that this callback has STARTED (pending -> running) and, if it ends without
        # an answer, that it ENDED (see the no_response rule of the arbiter).
        running=[(Output('url-init-running', 'data'), True, False)],
    )
    def url_init(request):
        if not isinstance(request, dict) or not request.get('epoch'):
            raise PreventUpdate
        search = request.get('search')
        search = search[:MAX_SEARCH_CHARS] if isinstance(search, str) else ''
        payload = url_state.make_init_payload(search, max_points_limit=limits.MAX_POINTS_LIMIT)
        return {'epoch': request['epoch'], 'payload': payload}

    # The arbiter: the ONLY writer of url-init, url-init-request and url-init-tick.disabled. The server answer is a State,
    # never an Input (an Input that is the output of a running callback would block this whole callback until it finished).
    app.clientside_callback(
        ClientsideFunction(namespace='url_restore', function_name='arbiter'),
        [Output('url-init', 'data'),
         Output('url-init-request', 'data'),
         Output('url-init-tick', 'disabled')],
        [Input('url-init-running', 'data'),
         Input('url-init-tick', 'n_intervals'),
         Input('url', 'search')],
        [State('url-init-config', 'data'),
         State('url-init-response', 'data')],
    )


    @app.callback(
        Output('url-plan-filters', 'data'),
        Input('url-plan-request', 'data'),
        [State('url-init', 'data'), State('derived-model-metadata', 'data')],
        prevent_initial_call=True,
    )
    def url_plan_filters(request, init, metadata):
        """
        Plan for the filters stage of the restore (validation against THIS browser's own loaded dataset).

        Nothing is written to any widget here. The answer echoes the request id, so the browser ignores an answer to a
        request it has since replaced, and carries the token and dataset key it was planned for.
        """
        if not isinstance(request, dict) or not request.get('token') or not request.get('id'):
            raise PreventUpdate
        from callbacks import callbacks_raven_labels as raven_labels
        from callbacks import callbacks_table as table
        payload = init.get('payload') if isinstance(init, dict) else None
        key = dataset_access.key_from_metadata(metadata)
        dataset = dataset_access.dataset_object(metadata)
        project = key[0] if key else None
        raven_ok = bool(project) and raven_labels.available(project)
        plan = url_plan.build_filters_plan(
            payload, request,
            dff=dataset.df if dataset is not None else None,
            metadata=dataset.metadata if dataset is not None else None,
            loaded_key=key, raven_available=raven_ok,
            annotators=raven_labels.project_annotators(project) if raven_ok else [],
            columns=url_plan.table_columns([c for c, _l, _v in table.ALL_COLUMNS], table.RAVEN_COLUMNS, raven_ok))
        plan['id'] = request['id']
        return plan

    @app.callback(
        Output('url-sync-result', 'data'),
        Input('url-sync-request', 'data'),
        State('derived-model-metadata', 'data'),
        prevent_initial_call=True,
    )
    def url_sync(request, metadata):
        """
        The query string for a settled snapshot of this browser's view (live URL or share link).

        All serialisation is ``url_state.rewrite_search`` (unrelated parameters preserved). The snapshot's dataset key
        must be the dataset THIS browser holds (its own Store), otherwise nothing is produced: a half-loaded or stale
        view is never turned into a link. Nothing is written to any widget.
        """
        if not isinstance(request, dict) or not request.get('id') or request.get('kind') not in ('live', 'share'):
            raise PreventUpdate
        snap = request.get('snapshot')
        base = {'id': request['id'], 'kind': request['kind']}
        key = dataset_access.key_from_metadata(metadata)
        dataset = dataset_access.dataset_object(metadata)
        if (not isinstance(snap, dict) or key is None or dataset is None
                or list(snap.get('key') or []) != [key[0], key[1], list(key[2])]):
            return {**base, 'status': 'dataset_mismatch'}
        search = url_plan.search_for_snapshot(snap, meta=dataset.metadata, dff=dataset.df, mode=request['kind'])
        return {**base, 'status': 'ok', 'search': search}
