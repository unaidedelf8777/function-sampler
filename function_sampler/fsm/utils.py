from transformers.convert_slow_tokenizer import import_protobuf
from transformers import PreTrainedTokenizer, AutoTokenizer
from function_sampler.cache import get_cache_dir
import os
import tempfile
import shutil


MODEL_PB2 = import_protobuf()

def trim_tokenizer_ascii(tokenizer: PreTrainedTokenizer):
    """
    Trims out tokens of the tokenizer that contain non-ascii characters.
    this leaves only tokens made of chars which are on the US standard keyboard.

    Main reason for this is to reduce the size of the tokenizer, which is useful when re-indexing the FSM,
    as the tokenizer will be smaller.
    """
    cache_dir_path = os.path.join(get_cache_dir(), f"{hash(tokenizer)}", "trimmed")
    if os.path.exists(cache_dir_path):
        return AutoTokenizer.from_pretrained(cache_dir_path)
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            tokenizer.save_pretrained(temp_dir, legacy_format=True) # legacy is easier to deal with
            print(os.listdir(temp_dir))

            with open(f"{temp_dir}/tokenizer.model", "rb") as f:
                m = MODEL_PB2.ModelProto.FromString(f.read())

                kept_parts = []
                for p in m.pieces:
                    if p.piece.strip().strip("\u2581").isascii(): ## llama models are wierd, but need the tokens with this char still.
                        kept_parts.append(p)

                kept_tokens = set([p.piece for p in kept_parts])
                i = 0
                while i < len(m.pieces):
                    if m.pieces[i].piece not in kept_tokens:
                        m.pieces.pop(i)
                    else:
                        i += 1
                
                if not os.path.exists(f"{get_cache_dir()}/{hash(tokenizer)}/trimmed/"):
                    os.makedirs(f"{get_cache_dir()}/{hash(tokenizer)}/trimmed/")

                
                with open(f"{get_cache_dir()}/{hash(tokenizer)}/trimmed/tokenizer.model", "wb") as f:
                    f.write(m.SerializeToString())
                
                ## copy all other tokenizer files from temp dir to cache dir
                for file in os.listdir(temp_dir):
                    if file != "tokenizer.model":
                        source_path = os.path.join(temp_dir, file)
                        target_path = os.path.join(cache_dir_path, file)
                        # Use shutil.copy instead of os.rename to copy file contents across filesystems
                        shutil.copy(source_path, target_path)
                
                return AutoTokenizer.from_pretrained(cache_dir_path)