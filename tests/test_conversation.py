"""Offline regression tests. No API requests, model downloads, or patient data."""
import asyncio
import copy
import os
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi import HTTPException
from fastapi.testclient import TestClient
from backend import app as api
from backend.conversation import Extracted, Observation, apply_evidence, decide, Decision
from backend.core import normalize_observations
from backend.dialogue import phrase_question

DEFINITIONS = {
    code: {'question_en': question, 'data_type': 'B', 'is_antecedent': False}
    for code, question in [('E_91', 'Have you had a fever?'), ('E_201', 'Do you have a cough?'),
                           ('E_148', 'Have you felt nauseated?'), ('E_97', 'Do you have a sore throat?'),
                           ('E_39', 'Have you felt confused?'), ('E_66', 'Do you have significant breathing difficulty?')]
}
CONDITIONS = {'URTI': {'symptoms': dict.fromkeys(DEFINITIONS)}}


def extracted(observations=None, **kwargs):
    return Extracted(observations=observations or [], urgent=False, urgent_quote='',
                     outside_scope=False, **kwargs)


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.state = {'answers': {}, 'asked': [], 'initial': None, 'pending_question': None,
                      'last_question': '', 'history': [], 'status': 'question'}

    def turn(self, data, text, ranking=None):
        updates = apply_evidence(self.state, data, text, DEFINITIONS)
        result = decide(self.state, data, text, DEFINITIONS, CONDITIONS,
                        ranking or [('URTI', .6)], updates)
        self.state['status'] = result.status
        return result

    def test_quote_whitespace_apostrophe_and_binary_case(self):
        observation = Observation(code=' e_91 ', values=['yes'], quote="I'm   feverish")
        self.assertEqual(normalize_observations([observation], DEFINITIONS,
                                               'I’m feverish today'), {'E_91': ['Yes']})
        self.assertEqual(normalize_observations([observation], DEFINITIONS, 'I feel well'), {})

    def test_detailed_message_preserves_all_recognized_findings(self):
        text = 'Since yesterday I have a fever, cough and nausea, with chills and sweating.'
        data = extracted([Observation(code=c, values=['Yes'], quote=q) for c, q in
                          [('E_91', 'fever'), ('E_201', 'cough'), ('E_148', 'nausea')]])
        result = self.turn(data, text)
        self.assertEqual(len(self.state['answers']), 3)
        self.assertNotIn(result.question_code, self.state['answers'])
        self.assertNotIn('main symptom', result.text)

    def test_empty_extraction_does_not_repeat_generic_question(self):
        first = self.turn(extracted(), 'Feeling chills and fever')
        second = self.turn(extracted(), 'I already said fever')
        self.assertNotEqual(first.text, second.text)
        self.assertEqual(second.status, 'uncertain')

    def test_unknown_correction_removes_evidence(self):
        self.state['answers'] = {'E_91': ['Yes']}
        self.turn(extracted(unknown_observations=[Observation(
            code='E_91', values=[], quote='not sure about fever')]), 'I am not sure about fever')
        self.assertNotIn('E_91', self.state['answers'])
        self.assertIn('E_91', self.state['unknown'])

    def test_negative_correction_overrides_previous_answer(self):
        self.state['answers'] = {'E_91': ['Yes']}
        self.turn(extracted([Observation(code='E_91', values=['No'], quote='no fever')]), 'Actually no fever')
        self.assertEqual(self.state['answers']['E_91'], ['No'])

    def test_summary_does_not_advance_or_reset_pending_question(self):
        self.state.update(answers={'E_91': ['Yes']}, pending_question='E_201')
        result = self.turn(extracted(intent='summary'), 'Summarize please')
        self.assertEqual(result.status, 'discussion')
        self.assertEqual(self.state['pending_question'], 'E_201')
        self.assertIn('fever', result.text)

    def test_malaria_mention_does_not_become_another_diagnosis(self):
        data = extracted([Observation(code='E_91', values=['Yes'], quote='fever')],
                         mentioned_condition='malaria', condition_quote='malaria')
        result = self.turn(data, 'Could my fever be malaria?', [('URTI', .99)])
        self.assertEqual(result.status, 'uncertain')
        self.assertIsNone(result.prediction)
        self.assertIn('E_91', self.state['answers'])

    def test_urgent_guidance_is_not_overridden_by_booking(self):
        self.state['status'] = 'urgent'
        result = self.turn(extracted(intent='booking'), 'Book an appointment')
        self.assertEqual(result.status, 'urgent')

    def test_phrasing_cap_falls_back_without_losing_question(self):
        client = SimpleNamespace(responses=SimpleNamespace(parse=AsyncMock()))
        result = asyncio.run(phrase_question(client, AsyncMock(side_effect=HTTPException(429)),
                             Decision('Have you had a fever?', rewrite=True), self.state))
        self.assertEqual(result, 'Have you had a fever?')
        client.responses.parse.assert_not_called()


class EndpointTests(unittest.TestCase):
    def setUp(self):
        api.SESSIONS.clear()
        self.remote = SimpleNamespace(responses=SimpleNamespace(parse=AsyncMock()))

        @asynccontextmanager
        async def fake_lifespan(app):
            yield

        replacements = {'definitions': DEFINITIONS, 'conditions': CONDITIONS, 'catalog': '{}',
                        'classifier': SimpleNamespace(config=SimpleNamespace(model_type='test')),
                        'client': self.remote}
        self.patchers = [patch.object(api, key, value, create=True) for key, value in replacements.items()]
        self.patchers += [patch.object(api.app.router, 'lifespan_context', fake_lifespan),
                          patch.object(api, 'reserve', AsyncMock()),
                          patch.object(api, 'predict', return_value=[('URTI', .6)]),
                          patch.dict(os.environ, {'NATURAL_REPLIES': 'false'})]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(api.app)
        self.addCleanup(self.client.close)
        self.sid = self.client.post('/sessions', json={'age': 23, 'sex': 'M'}).json()['session_id']

    def send(self, text, request_id=None):
        return self.client.post(f'/sessions/{self.sid}/messages', json={
            'text': text, 'request_id': request_id or str(uuid4())})

    def test_followup_after_assessment_and_request_deduplication(self):
        state = api.SESSIONS[self.sid]
        state.update(done=True, status='assessment', last_reply='One possibility is URTI.',
                     prediction='URTI', answers={'E_91': ['Yes']})
        self.remote.responses.parse.return_value = SimpleNamespace(output_parsed=extracted(intent='clarification'))
        request_id = str(uuid4())
        result = self.send('Why do you think that?', request_id)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()['status'], 'discussion')
        self.assertNotEqual(result.json()['reply'], 'One possibility is URTI.')
        self.assertEqual(result.json(), self.send('Why do you think that?', request_id).json())
        self.assertEqual(self.remote.responses.parse.call_count, 1)

    def test_direct_answer_uses_pending_question_only(self):
        state = api.SESSIONS[self.sid]
        state.update(asked=['E_66', 'E_201'], pending_question='E_66', last_question='Breathing difficulty?')
        result = self.send('Yes').json()
        self.assertEqual(result['status'], 'urgent')
        self.assertEqual(state['answers']['E_66'], ['Yes'])
        self.remote.responses.parse.assert_not_called()

    def test_failed_extraction_preserves_state(self):
        state = api.SESSIONS[self.sid]
        before = copy.deepcopy(state['answers'])
        self.remote.responses.parse.return_value = SimpleNamespace(output_parsed=None)
        self.assertEqual(self.send('I have a fever').status_code, 502)
        self.assertEqual(state['answers'], before)
        self.assertEqual(state['turns'], 0)

    def test_yes_after_assessment_does_not_answer_stale_question(self):
        state = api.SESSIONS[self.sid]
        state.update(status='assessment', asked=['E_66'], pending_question=None)
        self.remote.responses.parse.return_value = SimpleNamespace(output_parsed=extracted(intent='other'))
        self.assertEqual(self.send('Yes').json()['status'], 'discussion')
        self.assertNotIn('E_66', state['answers'])
        self.assertEqual(self.remote.responses.parse.call_count, 1)


if __name__ == '__main__':
    unittest.main()
