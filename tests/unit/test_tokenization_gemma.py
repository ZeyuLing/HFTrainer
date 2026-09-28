"""Golden contracts for Gemma's serialized Replace + Split BPE pipeline."""
from hftrainer.tokenization import LocalTokenizer


def tokenizer(normalizer=True):
    vocab={'<unk>':0,'a':1,'▁':2,'b':3,'▁b':4,'a▁b':5,'e':6,'\u0301':7,'é':8,'<0xCE>':9,'<0xBB>':10,'x':11,' ':12,'x ':13}
    return LocalTokenizer.from_buffer({
        'model':{'type':'BPE','vocab':vocab,'merges':[['▁','b'],['a','▁b'],['x',' ']],'unk_token':'<unk>','byte_fallback':True},
        'normalizer':{'type':'Replace','pattern':{'String':' '},'content':'▁'} if normalizer else None,
        'pre_tokenizer':{'type':'Split','pattern':{'String':' '},'behavior':'MergedWithPrevious','invert':False},
    })


def test_gemma_replace_preserves_spaces_and_bpe_can_cross_word_boundaries():
    t=tokenizer()
    assert t.encode('a b',add_special_tokens=False)==[5]
    assert t.encode(' a',add_special_tokens=False)==[2,1]
    assert t.encode('a  b',add_special_tokens=False)==[1,2,4]


def test_replace_does_not_add_unconfigured_unicode_normalization():
    t=tokenizer()
    assert t.encode('e\u0301',add_special_tokens=False)==[6,7]
    assert t.encode('é',add_special_tokens=False)==[8]


def test_bpe_byte_fallback_keeps_out_of_vocabulary_unicode():
    assert tokenizer().encode('λ',add_special_tokens=False)==[9,10]


def test_literal_split_merges_delimiter_with_previous_piece():
    assert tokenizer(False).encode('x a',add_special_tokens=False)==[13,1]
