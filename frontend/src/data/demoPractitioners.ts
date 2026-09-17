// Fictional names and profiles. No real clinical service or external image requests.
export const demoPractitioners = [
  { id: 'ada', name: 'Dr Ada Okafor', specialty: 'General medicine', initials: 'AO', color: '#d7ead0', description: 'General symptoms and next-step discussions.' },
  { id: 'tunde', name: 'Dr Tunde Bello', specialty: 'Family medicine', initials: 'TB', color: '#e4e9dc', description: 'Everyday health concerns and continuity of care.' },
  { id: 'amara', name: 'Dr Amara Eze', specialty: 'Internal medicine', initials: 'AE', color: '#d4e7e5', description: 'Adult health concerns and ongoing symptoms.' },
];
export type Appointment = { id: string; practitioner: string; date: string; time: string };
export const STORAGE_KEY = 'ddx.demoAppointments.v1';
export const slotTimes = ['09:00', '10:00', '11:00', '13:00', '14:00', '15:00'];
export function localDate(date = new Date()) {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
}
export function lastDate(now = new Date()) { const last = new Date(now); last.setDate(last.getDate() + 30); return localDate(last); }
export function validSlot(doctor: string, date: string, time: string, now = new Date()) {
  const chosen = new Date(`${date}T${time}:00`);
  return demoPractitioners.some(person => person.id === doctor) && /^\d{4}-\d{2}-\d{2}$/.test(date)
    && slotTimes.includes(time) && Number.isFinite(chosen.getTime()) && localDate(chosen) === date
    && date >= localDate(now) && date <= lastDate(now) && chosen > now && chosen.getDay() !== 0;
}
export function loadAppointments(storage: Pick<Storage, 'getItem'> = localStorage): Appointment[] {
  const parsed: unknown = JSON.parse(storage.getItem(STORAGE_KEY) || '[]');
  if (!Array.isArray(parsed)) return [];
  return parsed.filter((item): item is Appointment => item && typeof item.id === 'string'
    && typeof item.date === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(item.date)
    && slotTimes.includes(item.time) && demoPractitioners.some(person => person.id === item.practitioner));
}
