import time

from . import checking
from .http_client import LeanHTTPClient
from .proof_generation import generate_proof, generate_proof_batch
from .problem_structure import Attempt, TheoremProcessor


def generate_attempt(formal_statement, model, tokenizer, server_client: LeanHTTPClient, proof_type='', temperature=1, cot_proof="") -> Attempt:
	generation_start = time.time()
	proof = generate_proof(formal_statement, model, tokenizer, proof_type, temperature, cot_proof) 
	generation_time = time.time() - generation_start

	verification_start = time.time()
	success, message = checking.check_proof(proof, server_client)
	verification_time = time.time() - verification_start

	attempt = Attempt(success, proof, message, generation_time, verification_time)
	return attempt


def generate_initial_attempts(theorem_processor: TheoremProcessor, model, tokenizer, server_client: LeanHTTPClient, n_attempts=0, proof_type='', temperature=1, cot_proof="") -> None:
	if theorem_processor.has_solution():
		return
	formal_statement = '/-- ' + theorem_processor.informal_statement + '-/\n' + theorem_processor.formal_statement
	for _ in range(n_attempts):
		attempt = generate_attempt(formal_statement, model, tokenizer, server_client, proof_type, temperature, cot_proof)
		theorem_processor.add_initial_attempt(attempt)
		if theorem_processor.has_solution():
			break


def generate_initial_breakdown_attempts(theorem_processor: TheoremProcessor, model, tokenizer, server_client: LeanHTTPClient, n_attempts=0, proof_type='', temperature=1, cot_proof="") -> None:
	breakdown_attempt = theorem_processor.breakdown_attempts[0]
	if breakdown_attempt.has_successful_initial_breakdown_attempt():
		return
	formal_statement = '/-- ' + theorem_processor.informal_statement + '-/\n' + breakdown_attempt.theorem_manager.get_theorem_statement(include_lemmas_as_hypothesis=True, skip_unproven_lemmas=False)
	for _ in range(n_attempts):
		attempt = generate_attempt(formal_statement, model, tokenizer, server_client, proof_type, temperature, cot_proof)
		breakdown_attempt.add_initial_breakdown_attempt(attempt)
		if breakdown_attempt.has_successful_initial_breakdown_attempt():
			break


def generate_final_breakdown_attempts(theorem_processor: TheoremProcessor, model, tokenizer, server_client: LeanHTTPClient, n_attempts=0, proof_type='', temperature=1, cot_proof="") -> None:
	breakdown_attempt = theorem_processor.breakdown_attempts[0]
	if breakdown_attempt.has_solution():
		return
	formal_statement = breakdown_attempt.theorem_manager.get_theorem_statement(include_lemmas_as_hypothesis=True)
	for _ in range(n_attempts):
		if breakdown_attempt.has_solution():
			break
		attempt = generate_attempt(formal_statement, model, tokenizer, server_client, proof_type, temperature, cot_proof)
		breakdown_attempt.add_final_breakdown_attempt(attempt)


# Starting here it's all batch generation
def generate_attempts_batch(formal_statement: str, model, tokenizer, server_client: LeanHTTPClient, cot_proof="", batch_size=0, header="") -> list[Attempt]:
	"""
	Generates a batch of proofs and verifies them serially.
	"""
	generation_start = time.time()
	proof_batch = generate_proof_batch(formal_statement, model, tokenizer, cot_proof, batch_size, header)
	generation_time = time.time() - generation_start

	average_generation_time = generation_time / len(proof_batch)

	attempt_batch = []
	for proof in proof_batch:
		verification_start = time.time()
		# Verification remains serial as per confirmation
		success, message = checking.check_proof(proof, server_client)
		verification_time = time.time() - verification_start
		attempt_batch.append(Attempt(success, proof, message, average_generation_time, verification_time))

	return attempt_batch


def generate_initial_attempts_batch(theorem_processor: TheoremProcessor, model, tokenizer, server_client: LeanHTTPClient, n_attempts=0, cot_proof="", header="") -> None:
	if theorem_processor.has_solution() or n_attempts == 0:
		return
	
	formal_statement = '/-- ' + theorem_processor.informal_statement + '-/\n' + theorem_processor.formal_statement
	
	attempt_batch = generate_attempts_batch(formal_statement, model, tokenizer, server_client, cot_proof, n_attempts, header)
	
	for attempt in attempt_batch:
		theorem_processor.add_initial_attempt(attempt)
		# If a successful proof is found, stop adding attempts from this batch.
		if attempt.success:
			break


def generate_initial_breakdown_attempts_batch(theorem_processor: TheoremProcessor, model, tokenizer, server_client: LeanHTTPClient, n_attempts=0, cot_proof="", header="") -> None:
	if n_attempts == 0:
		return
	breakdown_attempt = theorem_processor.breakdown_attempts[0]
	if breakdown_attempt.has_successful_initial_breakdown_attempt():
		return

	formal_statement = '/-- ' + theorem_processor.informal_statement + '-/\n' + breakdown_attempt.theorem_manager.get_theorem_statement(include_lemmas_as_hypothesis=True, skip_unproven_lemmas=False)

	attempt_batch = generate_attempts_batch(formal_statement, model, tokenizer, server_client, cot_proof, n_attempts, header)
	
	for attempt in attempt_batch:
		breakdown_attempt.add_initial_breakdown_attempt(attempt)
		# If a successful proof is found, stop adding attempts from this batch.
		if attempt.success:
			break