"""Tests for BART multimodal processor helpers."""

import torch


def test_text_data_parser_handles_empty_inputs():
    from rh_custom_models_plugin.bart.bart import TextDataParser

    parser = TextDataParser()

    assert parser._parse_text_data("").data == [""]
    assert parser._parse_text_data([]).data == [""]


def test_create_encoder_prompt_uses_placeholder_for_decoder_tokens():
    from rh_custom_models_plugin.bart.bart import BartMultiModalProcessor

    processor = BartMultiModalProcessor.__new__(BartMultiModalProcessor)

    assert processor.create_encoder_prompt([0, 1, 2], {"texts": ["encoder text"]}) == [
        0
    ]


def test_apply_hf_processor_main_tokenizes_encoder_text():
    from rh_custom_models_plugin.bart.bart import (
        BartMultiModalProcessor,
        TextDataParser,
    )

    class FakeTokenizer:
        def __call__(self, text, return_tensors="pt", add_special_tokens=True):
            assert text == "encoder text"
            assert not add_special_tokens
            return {"input_ids": torch.tensor([[11, 12, 13]])}

    class FakeInfo:
        def get_tokenizer(self):
            return FakeTokenizer()

    processor = BartMultiModalProcessor.__new__(BartMultiModalProcessor)
    processor.info = FakeInfo()
    mm_items = TextDataParser().parse_mm_data({"text": "encoder text"})

    out = processor._apply_hf_processor_main(mm_items, {})

    assert torch.equal(out["encoder_input_ids"], torch.tensor([[11, 12, 13]]))
