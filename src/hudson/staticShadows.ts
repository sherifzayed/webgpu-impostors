export const SHADOW_CASTER_LAYER = 1;

let pending = true;

export function requestShadowUpdate() {
  pending = true;
}

/** Returns true once per request; the sun calls it every frame. */
export function consumeShadowUpdate(): boolean {
  const wasPending = pending;
  pending = false;
  return wasPending;
}
