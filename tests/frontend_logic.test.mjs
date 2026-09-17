import test from 'node:test';
import assert from 'node:assert/strict';
import { validSlot, loadAppointments, localDate } from '../frontend/src/data/demoPractitioners.ts';

const now = new Date('2026-09-16T08:00:00');
test('accepts a future appointment with a known practitioner', () => {
  assert.equal(validSlot('ada', '2026-09-16', '09:00', now), true);
});
test('rejects past, Sunday, distant and unknown slots', () => {
  for (const [doctor, date, time] of [
    ['ada', '2026-09-15', '09:00'], ['ada', '2026-09-20', '09:00'],
    ['ada', '2026-11-01', '09:00'], ['unknown', '2026-09-16', '09:00'],
    ['ada', '2026-09-16', '03:00'], ['ada', '2026-02-30', '09:00'],
  ]) assert.equal(validSlot(doctor, date, time, now), false);
});
test('loads saved bookings and filters malformed entries', () => {
  const booking = { id: 'example', practitioner: 'ada', date: localDate(now), time: '09:00' };
  assert.deepEqual(loadAppointments({ getItem: () => JSON.stringify([booking, null, {}]) }), [booking]);
  assert.deepEqual(loadAppointments({ getItem: () => null }), []);
});
test('invalid storage is surfaced to the interface for error handling', () => {
  assert.throws(() => loadAppointments({ getItem: () => 'broken JSON' }));
});
