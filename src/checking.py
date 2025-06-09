import time
import re

import http_client, formatting

def check_repl_status(server_client: http_client.LeanHTTPClient) -> tuple[bool, str]:
	if server_client is None:
		return False, "Server client is None"
	
	success, status = server_client.get_status()
	if not success:
		return False, "Failed to get server status"
	elif not status['ready']:
		print("REPL not ready, reinitializing...")
		success, status = server_client.reinitialize_repl()
		if not success:
			return False, "Failed to reinitialize REPL"
	return True, "REPL is ready"

def check_proof(proof, server_client: http_client.LeanHTTPClient, timeout=20, ignore_sorries=False):
	if server_client is None:
		return False, "Server client is None"
	
	if not proof:
		return False, "Proof provided was an empty string (probably a generation error)"

	success, message = check_repl_status(server_client)
	if not success:
		return False, message
		
	success, response = server_client.check_theorem(proof, timeout)
	if not success:
		return False, response
		
	if not ignore_sorries:
		if 'sorries' in response:
			return False, "Proof contains sorries"

	correct = True
	if 'messages' in response:
		for msg in response['messages']:
			if msg['severity'] == 'error':
				correct = False
				return False, response['messages']
	if correct:
		return True, "Proof verified successfully"
		

def check_lemmas_syntax(theorem_with_proofs, server_client: http_client.LeanHTTPClient):
	success, message = check_proof(proof=theorem_with_proofs, server_client=server_client, ignore_sorries=True)
	return success, message


def check_one_lemma_syntax(formal_statement, lemma, server_client: http_client.LeanHTTPClient):
	proof_with_sorries = formal_statement + "\n  " + lemma + " sorry" + "\n  sorry"
	success, message = check_proof(proof=proof_with_sorries, server_client=server_client, ignore_sorries=True)
	return success, message


def _find_subscript_l(text):
	# All subscript digits
	subscript_digits = "₀₁₂₃₄₅₆₇₈₉"
		
	# Check for each possible l₀ through l₉
	for i, subscript in enumerate(subscript_digits):
		if f"l{subscript}" in text:
			return f"l{subscript}"
		
	# If no subscript l found
	return ''


def get_unused_lemmas_from_proof(proof, server_client: http_client.LeanHTTPClient, timeout=20) -> list[str]:
	unused_lemmas = []
	success, response = server_client.check_theorem(proof, timeout)
	if not success or 'messages' not in response:
		return unused_lemmas
	for message in response['messages']:
		unused_lemmas.append(_find_subscript_l(message['data']))
	return unused_lemmas


def check_all_unchecked_proofs(theorems_dict: dict, server_client: http_client.LeanHTTPClient) -> None:
	i = 0
	for theorem in theorems_dict.values():
		if theorem['solution']:
			continue
		for attempt in theorem['initial_attempts']:
			if attempt['message'] == "Failed to get server status":
				i += 1
				print(i)
				proof = attempt['proof']

				verification_start = time.time()
				success, message = check_proof(proof, server_client)
				verification_time = time.time() - verification_start

				attempt['message'] = message
				attempt['verification_time'] = verification_time

				if success:
					print("SUCCESS")
					attempt['success'] = True
					theorem['solution'] = proof
					break


def validate_lemmas(formal_statement: str, lemmas: str, server_client: http_client.LeanHTTPClient, header=""):
	lemma_pattern = re.compile(r"^(have l)([₀₁₂₃₄₅₆₇₈₉]+)( : .*)$")
	processed_lines_for_this_problem = []
	renamed_have_lemma_index = 0
		
	for current_lemma_line in lemmas.splitlines():
		if not current_lemma_line.strip():
			continue
			
		success, msg = check_one_lemma_syntax(header[14:]+'\n'+formal_statement, current_lemma_line, server_client)
		
		print(current_lemma_line, success) 
		
		if success:
			match = lemma_pattern.match(current_lemma_line)
			
			if match:
				prefix = match.group(1)
				suffix_including_colon = match.group(3)
				
				new_subscript_str = formatting.to_subscript(renamed_have_lemma_index)
				
				renamed_lemma = f"{prefix}{new_subscript_str}{suffix_including_colon}"
				processed_lines_for_this_problem.append(renamed_lemma)
				
				renamed_have_lemma_index += 1
			else:
				processed_lines_for_this_problem.append(current_lemma_line)

	return '\n'.join(processed_lines_for_this_problem)