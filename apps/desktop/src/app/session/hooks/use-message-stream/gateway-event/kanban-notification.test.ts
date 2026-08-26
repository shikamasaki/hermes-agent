import { beforeEach, describe, expect, it, vi } from 'vitest'

const { notify, request } = vi.hoisted(() => ({
  request: vi.fn(async () => ({})),
  notify: vi.fn()
}))

vi.mock('@/store/gateway', () => ({
  $gateway: { get: () => ({ request }) }
}))
vi.mock('@/store/notifications', () => ({ notify }))

import { handleLifecycleEvent } from './lifecycle'

const card = {
  type: 'kanban.notification',
  session_id: 'bot-chat',
  payload: {
    board: 'default',
    delivery_key: 'chief:default:7',
    event_kind: 'blocked',
    outbox_id: 42,
    reason: 'needs input',
    task_id: 't_42'
  }
}

function ctx(event: { payload: unknown; session_id: string; type: string } = card, active = true) {
  return {
    deps: {},
    event,
    payload: event.payload,
    fromActiveSource: () => true,
    isActiveEvent: active
  } as any
}

describe('passive Kanban gateway events', () => {
  beforeEach(() => {
    request.mockClear()
    notify.mockClear()
  })

  it('renders an active canonical-session card and ACKs after presentation', () => {
    expect(handleLifecycleEvent(ctx())).toBe(true)
    expect(notify).toHaveBeenCalledWith(
      expect.objectContaining({
        id: 'chief:default:7',
        kind: 'warning',
        message: '[default] t_42 — needs input'
      })
    )
    expect(request).toHaveBeenCalledWith('kanban.notifications.ack', {
      surface: 'desktop',
      board: 'default',
      outbox_id: 42,
      delivery_key: 'chief:default:7'
    })
  })

  it('leaves a background-session card pending and does not paint it', () => {
    expect(handleLifecycleEvent(ctx(card, false))).toBe(true)
    expect(notify).not.toHaveBeenCalled()
    expect(request).not.toHaveBeenCalled()
  })

  it('subscribes the exact active session without consuming session.info', () => {
    const event = { type: 'session.info', session_id: 'bot-chat', payload: {} }
    expect(handleLifecycleEvent(ctx(event, true))).toBe(false)
    expect(request).toHaveBeenCalledWith('kanban.notifications.subscribe', {
      surface: 'desktop',
      session_id: 'bot-chat'
    })
  })
})
