# Manual checks (no browser automation required)

Keep both servers running; open the frontend yourself. Use fictional examples.
Real text and recorded-audio requests use your configured API quota.

## Conversation

- Start a new session; submit a detailed fictional fever/chills description.
- Confirm the bot does not repeatedly ask for the main symptom.
- Correct a symptom with an explicit negative; ask for a summary.
- Say you are unsure of a previously reported finding; check it is not treated as No.
- Ask for clarification after an assessment; confirm input remains available.
- Ask explicitly about malaria; confirm no malaria diagnosis is claimed with the
  existing dataset. The test is not proof of broad unsupported-condition detection.

## Voice

- Allow microphone access. Enter voice mode; typing should become read-only.
- In live-caption mode, speak and check the transcript before pressing Send.
- Check that the mic pauses, the reply is spoken automatically, and listening resumes.
- Mute, unmute, and End voice mode. Confirm End releases the mic and restores typing.
- If browser speech fails, check the visible error and retry Read aloud or listening.
- Switch to recorded audio while voice mode is off. Speak and press Send; verify
  transcription appears and is submitted. A 30-second recording stops and waits.
- Deny mic permission; verify the app shows recovery instructions and text remains
  available after ending voice mode.
- Change tabs and open appointments; verify voice mode stops.

## Appointments

- Select a fictional practitioner, a future weekday and a time; confirm the demo.
- Reload: the booking should remain. Try the same slot: it should be unavailable.
- Cancel and confirm the slot becomes available again.
- Try a past date, Sunday, or no time: confirmation must not be possible.
- Block browser storage: the app must not claim to have saved a booking.
- Check the layout at a narrow phone window and use Tab/Enter to operate controls.

## Before presenting

Verify MODEL_PATH names the model actually loaded. Keep the backend terminal
running, start a fresh session after restarting, and confirm microphone/audio
permissions on the exact browser and device used for the demonstration.
