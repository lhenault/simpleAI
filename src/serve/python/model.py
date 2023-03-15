from dataclasses import dataclass

@dataclass(unsafe_hash=True)
class LanguageModelServer:
    def complete(self, 
        prompt:             str='<|endoftext|>',
        suffix:             str='',
        max_tokens:         int=7, 
        temperature:        float=1.,
        top_p:              float=1.,
        n:                  int=1,
        stream:             bool=False,
        logprobs:           int=0,
        echo:               bool=False,
        stop:               str|list='',
        presence_penalty:   float=0.,
        frequence_penalty:  float=0.,
        best_of:            int=0,
        logit_bias:         dict={},
    ) -> str:
        # TODO : implements method for your LLM
        return 'QWERTYUIOP\nASDFGHJKL\nZXCVBNM'