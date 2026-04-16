/**
 * Normalizes a raw landmark Float32Array[84] for two hands.
 *
 * Layout: [hand0_lm0_x, hand0_lm0_y, hand0_lm1_x, hand0_lm1_y, ..., hand1_lm20_x, hand1_lm20_y]
 *
 * Step 1 (translation): subtract landmark[0] (wrist) from all points.
 * Step 2 (scale): divide by the Euclidean distance between landmark[0] and landmark[9].
 *
 * If the distance is zero (hand absent), returns a Float32Array of 84 zeros.
 *
 * Requisitos: 1.2, 1.3, 1.4
 */
export function normalize(raw: Float32Array): Float32Array {
  const SIZE = 84;
  const result = new Float32Array(SIZE);

  // landmark[0] is at indices [0, 1], landmark[9] is at indices [18, 19]
  const wristX = raw[0];
  const wristY = raw[1];
  const lm9X = raw[18];
  const lm9Y = raw[19];

  // After translation, lm9 relative to wrist
  const dx = lm9X - wristX;
  const dy = lm9Y - wristY;
  const distance = Math.sqrt(dx * dx + dy * dy);

  // If distance is zero (hand absent), return zeros
  if (distance === 0) {
    return result; // already all zeros
  }

  for (let i = 0; i < SIZE; i += 2) {
    result[i] = (raw[i] - wristX) / distance;
    result[i + 1] = (raw[i + 1] - wristY) / distance;
  }

  return result;
}
