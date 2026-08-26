import type { HermesSkin } from '@hermes/shared/skin'

import { $gateway } from '@/store/gateway'
import {
  notifyCronChanged,
  notifyPairingChanged,
  notifyPetChanged,
  notifyPlatformsChanged,
  notifySessionsChanged,
  type PetChangeMeta,
  setChangeEventsAvailable
} from '@/store/live-sync'
import { notify } from '@/store/notifications'
import { dropSessionState, unbindTileRuntime } from '@/store/session-states'
// Leaf import (not the `@/themes` barrel) to avoid pulling the ThemeProvider
// module graph into the gateway event hot path.
import { ingestBackendSkin } from '@/themes/backend-sync'

import type { GatewayEventContext } from './types'

/** gateway.ready / skin.changed / change-watcher broadcasts / session.reclaimed. */
export function handleLifecycleEvent(ctx: GatewayEventContext): boolean {
  const { deps, event, payload, fromActiveSource, isActiveEvent } = ctx

  if (event.type === 'kanban.notification') {
    const card = payload as
      | {
          board?: string
          delivery_key?: string
          event_kind?: string
          outbox_id?: number
          reason?: string
          status?: string
          summary?: string
          task_id?: string
          task_title?: string
        }
      | undefined

    // Only the exact active canonical session may consume/ACK its card. A
    // background-profile or non-Bot-Chat window leaves the durable row pending
    // for replay when the owner opens that session.
    if (fromActiveSource() && isActiveEvent && card?.delivery_key && card.board && card.outbox_id) {
      const detail = card.summary || card.reason || card.status
      const kind = card.event_kind === 'completed' ? 'success' : card.event_kind === 'blocked' ? 'warning' : 'info'
      notify({
        id: card.delivery_key,
        kind,
        title: `Kanban · ${card.event_kind || 'update'}`,
        message: `[${card.board}] ${card.task_title || card.task_id || 'task'}${card.task_title && card.task_id ? ` (${card.task_id})` : ''}${detail ? ` — ${detail}` : ''}`,
        durationMs: kind === 'success' ? 8_000 : 0
      })
      void $gateway.get()?.request('kanban.notifications.ack', {
          surface: 'desktop',
          board: card.board,
          outbox_id: card.outbox_id,
          delivery_key: card.delivery_key
        })
        .catch(() => undefined)
    }

    return true
  }

  if (event.type === 'gateway.ready') {
    // Seed the active skin into the desktop theme registry without applying,
    // so a fresh connect never overrides the user's persisted desktop theme.
    ingestBackendSkin((payload as { skin?: HermesSkin } | undefined)?.skin, { apply: false })
    // Backends with the change watcher broadcast pet/cron/sessions change
    // events; consumers demote their legacy polls to slow backstops.
    setChangeEventsAvailable(Boolean((payload as { change_events?: boolean } | undefined)?.change_events))

    return true
  }

  if (event.type === 'session.info' && event.session_id && isActiveEvent && fromActiveSource()) {
    void $gateway.get()?.request('kanban.notifications.subscribe', {
        surface: 'desktop',
        session_id: event.session_id
      })
      .catch(() => undefined)

    return false
  }

  if (event.type === 'skin.changed') {
    // A runtime skin switch (Hermes activating an authored skin, or `/skin`
    // on another surface). Only the active source+profile's change repaints.
    if (fromActiveSource()) {
      ingestBackendSkin(payload as HermesSkin | undefined, { apply: true })
    }

    return true
  }

  if (
    event.type === 'pet.changed' ||
    event.type === 'cron.changed' ||
    event.type === 'sessions.changed' ||
    event.type === 'platforms.changed' ||
    event.type === 'pairing.changed'
  ) {
    // Change-watcher broadcasts (server._broadcast_watched_changes): the
    // backend's on-disk signature moved. Route to the live-sync ticks the
    // former pollers now subscribe to. Only the active source+profile's
    // changes apply — background profile sockets (and other connections'
    // gateways) watch their own homes.
    if (fromActiveSource()) {
      if (event.type === 'pet.changed') {
        notifyPetChanged(payload as PetChangeMeta | undefined)
      } else if (event.type === 'cron.changed') {
        notifyCronChanged()
      } else if (event.type === 'platforms.changed') {
        notifyPlatformsChanged()
      } else if (event.type === 'pairing.changed') {
        notifyPairingChanged()
      } else {
        notifySessionsChanged()
      }
    }

    return true
  }

  if (event.type === 'session.reclaimed') {
    // The backend reclaimed a live session we may still be holding (idle
    // TTL, LRU cap, or the WS-orphan reap). Without this the runtime id
    // stays cached until something fails against it, which reads as the
    // session vanishing rather than being reclaimed. Drop the cached state
    // now — the stored row is untouched, so the sidebar keeps the
    // conversation and reopening it resumes from the DB.
    const reclaimedRuntimeId = String((payload as { session_id?: string } | undefined)?.session_id ?? '')

    if (reclaimedRuntimeId) {
      dropSessionState(reclaimedRuntimeId)
      // A tile bound to the reclaimed runtime would otherwise render an
      // empty transcript forever: its view reads $sessionStates[runtime]
      // (just dropped) and its resume effect is gated on !runtimeId, so a
      // bound tile never re-resumes (#82620). Unbind it so the effect
      // refires against the intact stored session — and purge the wiring
      // cache's entry, or resumeTile's warm path would hand the dead
      // runtime straight back instead of cold-resuming a live one.
      unbindTileRuntime(reclaimedRuntimeId)
      deps.sessionStateByRuntimeIdRef.current.delete(reclaimedRuntimeId)
    }

    // The row's ended_at moved, so refresh the lists that render it.
    notifySessionsChanged()

    return true
  }

  return false
}
