from . import slm

def generate_basic_proof(formal_statement, model, tokenizer, temperature=1):
	prompt = """Complete the following Lean 4 code:
‘‘‘lean4
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

""" + formal_statement + '\n'
		
	proof = slm.SLM_proof(prompt, model, tokenizer, temperature)
	return proof


def _add_cot_proof_comment(cot_proof):
	if not cot_proof:
		return ""
	proof_lines = cot_proof.splitlines()
	for i in range(len(proof_lines)):
		proof_lines[i] = '  ' + proof_lines[i]
	return '  /-\n' + '\n'.join(proof_lines) + '\n  -/'


def generate_cot_proof(formal_statement, model, tokenizer, temperature=1, cot_proof=""):
	prompt = """Complete the following Lean 4 code with explanatory comments preceding each line of code:
‘‘‘lean4
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

""" + formal_statement + '\n' + _add_cot_proof_comment(cot_proof)
		
	proof = slm.SLM_proof(prompt, model, tokenizer, temperature)
	return proof


def generate_rtg_proof(formal_statement):
	tactics = ['linarith', 'simp', 'norm_num', 'nlinarith', 'simp_all', 'ring', 'field_simp', 'assumption', 'omega', 'rfl', 'norm_cast', 'constructor', 'decide', 'aesop']
	proof = formal_statement
	for tactic in tactics:
		proof += f"\n  try {tactic}"
	return proof


def _format_proof(proof):
	# NEEDS TESTING
	# Format proof
	proof_lines = proof.splitlines()
	formatted_proof = '\n'.join(proof_lines[3:-1])

	# Return formatted proof
	return formatted_proof


def generate_proof(formal_statement, model, tokenizer, proof_type='', temperature=1, cot_proof=""):
	try:
		if proof_type == 'cot':
			proof = generate_cot_proof(formal_statement, model, tokenizer, temperature, cot_proof)
			proof = _format_proof(proof)
		else:
			proof = generate_basic_proof(formal_statement, model, tokenizer, temperature)
			proof = _format_proof(proof)

		return proof
		
	except Exception as e:
		print(f"Error found: {e}")
		return ""
	

def generate_proof_batch(formal_statement: str, model, tokenizer, cot_proof="", batch_size=0, header=""):
	standard_header = """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

"""
	prompt = """Complete the following Lean 4 code with explanatory comments preceding each line of code:
‘‘‘lean4\n""" + (header if header else standard_header) + formal_statement + '\n' + _add_cot_proof_comment(cot_proof)

	prompt_batch = [prompt] * batch_size
	try:
		proof_batch = slm.SLM_proof_batch(prompt_batch, model, tokenizer)
		return [_format_proof(proof) for proof in proof_batch]
	except Exception as e:
		print(f"Error found: {e}")
		return [""] * batch_size