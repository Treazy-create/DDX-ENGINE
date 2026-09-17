import React, { useEffect, useState } from 'react';
import { ArrowLeft, CalendarDays, Check, Stethoscope, Trash2 } from 'lucide-react';
import { demoPractitioners, lastDate, loadAppointments, localDate, slotTimes, STORAGE_KEY, validSlot, type Appointment } from '../data/demoPractitioners';

export function AppointmentBooking({ onBack }: { onBack(): void }) {
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [doctor, setDoctor] = useState(demoPractitioners[0].id);
  const [date, setDate] = useState(localDate());
  const [time, setTime] = useState('');
  const [error, setError] = useState('');
  const [confirmation, setConfirmation] = useState<Appointment | null>(null);
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const refresh = () => { try { setAppointments(loadAppointments()); } catch { setError('Saved bookings could not be read. Browser storage may be unavailable.'); } };
    refresh(); window.addEventListener('storage', refresh);
    const timer = setInterval(() => setNow(new Date()), 60000);
    return () => { clearInterval(timer); window.removeEventListener('storage', refresh); };
  }, []);
  const person = demoPractitioners.find(item => item.id === doctor)!;
  const taken = (slot: string) => appointments.some(item => item.practitioner === doctor && item.date === date && item.time === slot);
  function save(next: Appointment[]) {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(next)); setAppointments(next); setError(''); return true; }
    catch { setError('This browser could not save the booking. No confirmation was created.'); return false; }
  }
  function book(event: React.FormEvent) {
    event.preventDefault();
    if (!validSlot(doctor, date, time)) { setError('Choose an available future date and time.'); return; }
    try {
      const latest = loadAppointments();
      if (latest.some(item => item.practitioner === doctor && item.date === date && item.time === time)) {
        setAppointments(latest); setError('That time is already booked in this browser. Choose another.'); return;
      }
      const appointment = { id: crypto.randomUUID(), practitioner: doctor, date, time };
      if (save([...latest, appointment])) { setConfirmation(appointment); setTime(''); }
    } catch { setError('Saved bookings could not be read. Please enable browser storage.'); }
  }
  function cancel(id: string) {
    try { if (save(loadAppointments().filter(item => item.id !== id))) setConfirmation(null); }
    catch { setError('The booking could not be cancelled. Please try again.'); }
  }
  return <section className="booking-page" aria-labelledby="booking-title">
    <button className="back-link" onClick={onBack}><ArrowLeft size={17} /> Back to conversation</button>
    <h2 id="booking-title">Make room for a conversation.</h2>
    <p className="booking-intro">Choose a practitioner and a time that works for you.</p>
    <p className="demo-notice"><CalendarDays size={18} /> Demonstration only. These profiles are fictional; no real appointment is arranged.</p>
    {error && <p role="alert" className="error">{error}</p>}
    {confirmation && <div className="booking-confirmation" role="status">
      <Check size={24} /><div><strong>Demo appointment saved</strong><p>{demoPractitioners.find(item => item.id === confirmation.practitioner)?.name} · {confirmation.date} at {confirmation.time}</p>
      <small>Saved on this browser only. No practitioner has been contacted.</small></div>
    </div>}
    <form onSubmit={book}>
      <fieldset className="practitioner-list"><legend>Choose a practitioner</legend>
        {demoPractitioners.map(item => <label key={item.id} className={'practitioner ' + (doctor === item.id ? 'selected' : '')}>
          <input type="radio" name="practitioner" value={item.id} checked={doctor === item.id} onChange={() => { setDoctor(item.id); setTime(''); setConfirmation(null); }} />
          <span className="practitioner-avatar" style={{ background: item.color }} aria-hidden="true"><Stethoscope size={30} /><small>{item.initials}</small></span>
          <span><strong>{item.name}</strong><span>{item.specialty}</span><small>{item.description}</small></span>
        </label>)}
      </fieldset>
      <div className="booking-schedule">
        <label className="date-field">Choose a date<input type="date" required min={localDate(now)} max={lastDate(now)} value={date}
          onChange={event => { setDate(event.target.value); setTime(''); setConfirmation(null); }} /></label>
        <fieldset className="time-field"><legend>Choose a time</legend><div className="time-slots">
          {slotTimes.map(slot => <button key={slot} type="button" aria-pressed={time === slot}
            disabled={!validSlot(doctor, date, slot, now) || taken(slot)} onClick={() => setTime(slot)}>{slot}</button>)}
        </div><small>Times use your device’s local timezone. No Sunday slots.</small></fieldset>
      </div>
      <div className="booking-summary"><p>{person.name}<span>{date || 'Choose a date'} · {time || 'Choose a time'}</span></p>
        <button className="primary" disabled={!time || !validSlot(doctor, date, time, now) || taken(time)}>Confirm demo appointment</button></div>
    </form>
    <section className="saved-appointments" aria-labelledby="saved-title"><h3 id="saved-title">Your demo appointments</h3>
      {!appointments.length && <p>No appointments saved yet.</p>}
      {appointments.map(item => <div className="saved-appointment" key={item.id}>
        <CalendarDays size={20} /><p><strong>{demoPractitioners.find(person => person.id === item.practitioner)?.name}</strong><span>{item.date} · {item.time}</span></p>
        <button className="back-link" onClick={() => cancel(item.id)} aria-label={`Cancel ${item.date} ${item.time} appointment`}><Trash2 size={16} /> Cancel</button>
      </div>)}
    </section>
  </section>;
}
