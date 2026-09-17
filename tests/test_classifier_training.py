"""Tiny random BERT tests: real training/checkpoint code, no pretrained downloads."""
import random
import shutil
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import torch
from transformers import BertConfig, BertForSequenceClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from classifier_data import case_text
from classifier_training import configure_training, save_checkpoint, step, train, training_mode


class ClassifierTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.model = BertForSequenceClassification(BertConfig(
            vocab_size=32, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=32, num_labels=2, max_position_embeddings=16))
        self.optimizer = configure_training(self.model, 1)
        self.batch = {'input_ids': torch.tensor([[2, 3, 4, 0], [2, 8, 4, 0]]),
                      'attention_mask': torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]]),
                      'labels': torch.tensor([0, 1])}

    def test_only_selected_backbone_layers_train(self):
        frozen = self.model.bert.embeddings.word_embeddings.weight.detach().clone()
        head = self.model.classifier.weight.detach().clone()
        training_mode(self.model, 1)
        loss = step(self.model, self.optimizer, self.batch, 'cpu')
        self.assertTrue(loss > 0)
        self.assertTrue(torch.equal(frozen, self.model.bert.embeddings.word_embeddings.weight))
        self.assertFalse(torch.equal(head, self.model.classifier.weight))

    def test_resume_matches_next_uninterrupted_update(self):
        folder = Path(__file__).parent / ('checkpoint_test_' + uuid4().hex)
        folder.mkdir()
        self.addCleanup(shutil.rmtree, folder)
        training_mode(self.model, 1)
        step(self.model, self.optimizer, self.batch, 'cpu')
        save_checkpoint(folder, self.model, self.optimizer, {'epoch': 1, 'step': 1})
        self.assertFalse((folder / 'last_checkpoint.tmp').exists())
        step(self.model, self.optimizer, self.batch, 'cpu')
        expected = self.model.classifier.weight.detach().clone()
        saved = torch.load(folder / 'last_checkpoint.pt', weights_only=True)
        restored = BertForSequenceClassification(BertConfig.from_dict(saved['config']))
        restored.load_state_dict(saved['model'])
        optimizer = configure_training(restored, 1)
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng'])
        training_mode(restored, 1)
        step(restored, optimizer, self.batch, 'cpu')
        self.assertTrue(torch.equal(expected, restored.classifier.weight))

    def test_partial_history_is_reproducible_and_retains_initial(self):
        row = {'input_text': 'Age: 20 years.\nRecorded sex: male.\nDO NOT COPY LABEL',
               'initial_evidence': 'A', 'evidence': [
                   {'code': code, 'question': code + '?', 'answer': 'Yes'} for code in 'ABCDEF']}
        first = case_text(row, random.Random(1042), partial=True)
        self.assertEqual(first, case_text(row, random.Random(1042), partial=True))
        self.assertIn('A? Answer: Yes.', first)
        self.assertLess(len(first.splitlines()), 8)
        self.assertNotIn('LABEL', first)

    def test_one_epoch_writes_best_model_and_resume_checkpoint(self):
        folder = Path(__file__).parent / ('checkpoint_test_' + uuid4().hex)
        folder.mkdir()
        self.addCleanup(shutil.rmtree, folder)

        class TinyTokenizer:
            def __call__(self, texts, **kwargs):
                return {'input_ids': torch.tensor([[2, 3, 4]] * len(texts)),
                        'attention_mask': torch.ones(len(texts), 3, dtype=torch.long)}

            def save_pretrained(self, target):
                (target / 'tokenizer_test.txt').write_text('test fixture')

        rows = [{'input_text': 'Age: 20 years.\nRecorded sex: male.', 'label': label,
                 'initial_evidence': 'A', 'evidence': [{'code': 'A', 'question': 'Fever?', 'answer': 'Yes'}]}
                for label in [0, 1]]
        args = SimpleNamespace(seed=42, batch_size=2, max_length=8, layers=1, epochs=1, save_every=1)
        state = {'epoch': 1, 'step': 0, 'history': [], 'best_score': -1.}
        train(self.model, TinyTokenizer(), self.optimizer, rows, rows, args, folder, state, 'cpu')
        self.assertTrue((folder / 'best_model/config.json').exists())
        self.assertTrue((folder / 'training_report.json').exists())
        saved = torch.load(folder / 'last_checkpoint.pt', weights_only=True)
        self.assertEqual((saved['epoch'], saved['step']), (2, 0))


if __name__ == '__main__':
    unittest.main()
