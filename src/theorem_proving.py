import formatting, checking, attempt_generation
from http_client import LeanHTTPClient
from problem_structure import TheoremProcessor
from theorem_manager import ProofStructure
  
# Modified functions, deleted most and changed all to only work with CoT and assuming all previus (even unproven) lemmas for each next lemma, and never doing final_theorem_breakdown. If you want to go back the old file it's in '/old'

def attempt_theorem_breakdown_proof(
	theorem_processor: TheoremProcessor,
	model,
	tokenizer,
	server_client: LeanHTTPClient,
	n_initial_breakdown_attempts=1,
	n_attempts_per_lemma=1,
	use_llm_cot=True,
):
	breakdown_attempt = theorem_processor.breakdown_attempts[0]
	attempt_generation.generate_initial_breakdown_attempts(
		theorem_processor, model, tokenizer, server_client,
		n_attempts=n_initial_breakdown_attempts,
		proof_type='cot',
		cot_proof=breakdown_attempt.proof_structure.final_proof if use_llm_cot else ""
	)
	if not breakdown_attempt.has_successful_initial_breakdown_attempt():
		return
	
	for lemma_name, lemma in breakdown_attempt.lemmas.items():
		attempt_generation.generate_initial_attempts(
			lemma, model, tokenizer, server_client, 
			n_attempts=n_attempts_per_lemma, 
			proof_type='cot',
			cot_proof=lemma.informal_proof if use_llm_cot else ""
		)
		if lemma.has_solution():
			breakdown_attempt.add_lemma_proof(lemma_name, lemma.solution)

	# Simply check if the proof is correct
	current_proof = breakdown_attempt.theorem_manager.get_theorem_with_proofs()
	success, _ = checking.check_proof(current_proof, server_client)
	if success:
		breakdown_attempt.solution = current_proof

	if breakdown_attempt.has_solution():
		theorem_processor.add_solution(breakdown_attempt.get_solution())


def attempt_theorem_breakdown_proof_no_checking(
	theorem_processor: TheoremProcessor,
	model,
	tokenizer,
	n_initial_breakdown_attempts=1,
	n_attempts_per_lemma=1,
	use_llm_cot=True,
):
	breakdown_attempt = theorem_processor.breakdown_attempts[0]
	attempt_generation.generate_initial_breakdown_attempts(
		theorem_processor, model, tokenizer,
		server_client=None,
		n_attempts=n_initial_breakdown_attempts,
		proof_type='cot',
		cot_proof=breakdown_attempt.proof_structure.final_proof if use_llm_cot else ""
	)
	# Do not skip if there's no successful initial breakdown attempt
	
	for lemma_name, lemma in breakdown_attempt.lemmas.items():
		attempt_generation.generate_initial_attempts(
			lemma, model, tokenizer,
			server_client=None, 
			n_attempts=n_attempts_per_lemma, 
			proof_type='cot',
			cot_proof=lemma.informal_proof if use_llm_cot else ""
		)
	# Do not check anything


def attempt_new_theorem_breakdown(
	theorem_processor: TheoremProcessor,
	proof_structure: ProofStructure,
	server_client: LeanHTTPClient,
	nl_proof="",
) -> None:
	if theorem_processor.has_solution():
		return
	if proof_structure is None:
		return
	try:
		breakdown_attempt = theorem_processor.add_breakdown_attempt(nl_proof, proof_structure)	
	except Exception as e:
		print(f"Error in generation with LLM (possibly with formatting): {e}")
		return
	
	for i, (lemma_name, lemma_entry) in enumerate(proof_structure.lemmas.items()):
		lemma = TheoremProcessor(
			formal_statement=breakdown_attempt.theorem_manager.get_lemma_as_theorem(i, skip_unproven_lemmas=False),
			informal_statement=lemma_entry.informal_statement,
			informal_proof=lemma_entry.informal_proof
		)
		breakdown_attempt.add_lemma(lemma_name, lemma)

	# If you delete everything from this line onwards you'll have the previous version of this function
	# Adding the lemma proofs to the breakdown attempt
	attempts = theorem_processor.initial_attempts
	formal_statement = theorem_processor.formal_statement

	lemmas_proofs_from_initial_attempts = formatting.map_target_lemmas_to_failed_proofs(
		[lemma.formal_statement for lemma in breakdown_attempt.proof_structure.lemmas.values()],
		[attempt.proof for attempt in attempts]
	)

	for lemma, proofs in lemmas_proofs_from_initial_attempts.items():
		lemma_name = lemma[5:7]
		for proof in proofs:
			lemma_proof = f"{formal_statement}\n  {lemma}\n{proof}\n  sorry"
			success, _ = checking.check_lemmas_syntax(lemma_proof, server_client)
			if success:
				lemma_proof_without_statement = f"{formal_statement}\n  -- {lemma}\n  -- Proof extracted from failed attempts\n{formatting.remove_two_leading_spaces_from_lines(proof)}"

				breakdown_attempt.add_lemma_proof(lemma_name, lemma_proof_without_statement)
				breakdown_attempt.lemmas[lemma_name].add_solution(lemma_proof_without_statement)

				break


def attempt_more_proofs(
	theorem_processor: TheoremProcessor,
	model,
	tokenizer,
	server_client: LeanHTTPClient,
	n_initial_attempts=0,
	n_initial_breakdown_attempts=0,
	n_attempts_per_lemma=0,
	cot_proof=""
	):
	if theorem_processor.has_solution():
		return
	
	attempt_generation.generate_initial_attempts(
		theorem_processor, model, tokenizer, server_client,
		n_attempts=n_initial_attempts,
		proof_type='cot', cot_proof=cot_proof,
	)
	
	for breakdown_attempt in theorem_processor.breakdown_attempts:
		if theorem_processor.has_solution():
			return
		
		# Attempt to prove theorem breakdown
		attempt_theorem_breakdown_proof(
			theorem_processor, model, tokenizer, server_client,
			n_initial_breakdown_attempts=n_initial_breakdown_attempts,
			n_attempts_per_lemma=n_attempts_per_lemma,
		)

		# Add proof to problem (if present)
		if breakdown_attempt.has_solution():
			theorem_processor.add_solution(breakdown_attempt.get_solution())


# Starting here we use batches
"""
def attempt_theorem_breakdown_proof_batch(
	theorem_processor: TheoremProcessor,
	model,
	tokenizer,
	server_client: LeanHTTPClient,
	n_initial_breakdown_attempts=1,
	n_attempts_per_lemma=1,
	use_llm_cot=True,
):
	breakdown_attempt = theorem_processor.breakdown_attempts[0]
	# Use the batch function for the initial breakdown attempt
	attempt_generation.generate_initial_breakdown_attempts_batch(
		theorem_processor, model, tokenizer, server_client,
		n_attempts=n_initial_breakdown_attempts,
		cot_proof=breakdown_attempt.proof_structure.final_proof if use_llm_cot else ""
	)
	if not breakdown_attempt.has_successful_initial_breakdown_attempt():
		return
	
	for lemma_name, lemma in breakdown_attempt.lemmas.items():
		# Use the batch function for each lemma
		attempt_generation.generate_initial_attempts_batch(
			lemma, model, tokenizer, server_client, 
			n_attempts=n_attempts_per_lemma, 
			cot_proof=lemma.informal_proof if use_llm_cot else ""
		)
		if lemma.has_solution():
			breakdown_attempt.add_lemma_proof(lemma_name, lemma.solution)

	# Simply check if the proof is correct
	current_proof = breakdown_attempt.theorem_manager.get_theorem_with_proofs()
	success, _ = checking.check_proof(current_proof, server_client)
	if success:
		breakdown_attempt.solution = current_proof

	if breakdown_attempt.has_solution():
		theorem_processor.add_solution(breakdown_attempt.get_solution())
"""


def attempt_theorem_breakdown_proof_batch_no_checking(
	theorem_processor: TheoremProcessor,
	model,
	tokenizer,
	n_initial_breakdown_attempts=1,
	n_attempts_per_lemma=1,
	use_llm_cot=True,
	header=""
):
	breakdown_attempt = theorem_processor.breakdown_attempts[0]
	# NOTE: Assuming generate_..._batch functions will correctly handle server_client=None 
	# by skipping the verification step.
	attempt_generation.generate_initial_breakdown_attempts_batch(
		theorem_processor, model, tokenizer,
		server_client=None,
		n_attempts=n_initial_breakdown_attempts,
		cot_proof=breakdown_attempt.proof_structure.final_proof if use_llm_cot else "",
		header=header
	)
	# Do not skip if there's no successful initial breakdown attempt
	
	for lemma_name, lemma in breakdown_attempt.lemmas.items():
		attempt_generation.generate_initial_attempts_batch(
			lemma, model, tokenizer,
			server_client=None, 
			n_attempts=n_attempts_per_lemma, 
			cot_proof=lemma.informal_proof if use_llm_cot else "",
			header=header
		)
	# Do not check anything


def attempt_more_proofs_batch(
	theorem_processor: TheoremProcessor,
	model,
	tokenizer,
	server_client: LeanHTTPClient,
	n_initial_attempts=0,
	n_initial_breakdown_attempts=0,
	n_attempts_per_lemma=0,
	use_llm_cot=True,
	cot_proof="",
	header="",
	):
	if theorem_processor.has_solution():
		return
	
	# Use the batch function for initial attempts
	attempt_generation.generate_initial_attempts_batch(
		theorem_processor, model, tokenizer, server_client,
		n_attempts=n_initial_attempts,
		cot_proof=cot_proof if use_llm_cot else "",
		header=header,
	)
	
	if theorem_processor.has_solution():
		return

	# Call the newly renamed batch version of this function
	attempt_theorem_breakdown_proof_batch_no_checking(
		theorem_processor, model, tokenizer,
		n_initial_breakdown_attempts=n_initial_breakdown_attempts,
		n_attempts_per_lemma=n_attempts_per_lemma,
		use_llm_cot=use_llm_cot,
		header=header,
	)

	breakdown_attempt = theorem_processor.breakdown_attempts[0]

	# Add proof to problem (if present)
	if breakdown_attempt.has_solution():
		theorem_processor.add_solution(breakdown_attempt.get_solution())