import torch
from transformers import TextIteratorStreamer
from threading import Thread, Event

def SLM_proof_stream(prompt: str, model, tokenizer, temperature=1):
    """
    Generates formal Lean proof steps using a specialized language model, with streaming output.

    Args:
        prompt: Input containing theorem and proof context
        model: Specialized language model
        tokenizer: Model tokenizer
    Yields:
        Generated tokens as they're produced
    
    Example usage:
        response = ''
            for token in SLM_proof_stream(prompt, model, tokenizer):
	        response += token
    """
    # Encode with attention mask
    encoded = tokenizer(
        prompt,
        return_tensors='pt',
    )

    # Move everything to CUDA
    inputs = {
        'input_ids': encoded['input_ids'].to('cuda'),
        'attention_mask': encoded['attention_mask'].to('cuda')
    }

    # Create streamer
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    # Generate in a separate thread
    generation_kwargs = dict(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        max_length=2048,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Skip the first chunk (input text)
    first = True
    for text in streamer:
        if first:
            yield text
            first = False
            continue
            
        if '--' in text:
            thread.join(timeout=0)
            break
        yield text


def SLM_proof(prompt: str, model, tokenizer, temperature=1) -> str:
    """
    Generates formal Lean proof steps using a specialized language model.

    Args:
        prompt: Input containing theorem and proof context
        model: Specialized language model
        tokenizer: Model tokenizer
    Returns:
        Generated Lean proof steps
    """
    # Encode with attention mask
    encoded = tokenizer(
        prompt,
        return_tensors='pt',
    )

    # Move everything to CUDA
    inputs = {
        'input_ids': encoded['input_ids'].to('cuda'),
        'attention_mask': encoded['attention_mask'].to('cuda')
    }

    # Generate
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        max_length=2048,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and return
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result


def SLM_proof_batch(prompts: list[str], model, tokenizer, temperature=1.0) -> list[str]:
    """
    Processes a batch of prompts with a hardcoded max_length of 2048.
    """
    # --- FIX ---
    # The tokenizer needs a padding token to handle batches of different lengths.
    # We set the padding token to be the same as the end-of-sequence token.
    # This is a standard practice for many autoregressive models.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # --- END FIX ---

    encoded = tokenizer(
        prompts, return_tensors='pt', padding=True, truncation=True
    )
    inputs = {k: v.to('cuda') for k, v in encoded.items()}
    
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        max_length=2048,
        # We also explicitly tell the model generation to use this pad_token_id
        pad_token_id=tokenizer.pad_token_id
    )
    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return results



def truncate_at_new_comment(incomplete_proof: str, complete_proof: str) -> str:
    """
    Truncates complete_proof at the first comment (line starting with two spaces followed by --)
    that doesn't appear in incomplete_proof.
    
    Args:
        incomplete_proof: The current state of the proof
        complete_proof: A complete version of the proof
        
    Returns:
        The truncated complete_proof
    """
    # Check if complete_proof is continuation of incomplete_proof
    if not complete_proof.startswith(incomplete_proof):
        raise ValueError("Incomplete proof is not a prefix of complete proof")

    # Get rest of proof and divide it into lines
    rest_of_proof = complete_proof[len(incomplete_proof):]
    rest_lines = rest_of_proof.splitlines()

    # Go through each line and check if it is a comment
    truncate_index = 0
    while truncate_index < len(rest_lines):
        line = rest_lines[truncate_index]
        if "--" in line:
            break
        truncate_index += 1

    # Concatenate proofs and return
    truncated_proof = incomplete_proof + '\n'.join(rest_lines[:truncate_index]) + '\n'
    return truncated_proof