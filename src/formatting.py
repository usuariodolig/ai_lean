from dataclasses import dataclass
import collections
from typing import List, Optional, Tuple, Dict
import re

@dataclass
class Hypothesis:
    raw_text: str
    name: Optional[str]  # For named hypotheses like h₁
    content: str


class TheoremParser:
    @staticmethod
    def parse(theorem_text: str) -> Tuple[str, List[Hypothesis], str]:
        stripped_theorem_text = TheoremParser._remove_trailing_newlines_and_by(theorem_text)
        clean_theorem_text = TheoremParser._remove_comment_lines(stripped_theorem_text)
        name, rest_of_text = TheoremParser._split_theorem_text(clean_theorem_text)
        hypotheses, conclusion = TheoremParser._parse_theorem(rest_of_text)
        
        return name, hypotheses, conclusion

    @staticmethod
    def extract_proof(theorem_text: Optional[str]) -> str:
        if not theorem_text:
            return None
        lines = theorem_text.split('\n')
        
        # Find the line containing ':= by'
        proof_start = -1
        for i, line in enumerate(lines):
            if ":= by" in line:
                proof_start = i
                break
                
        if proof_start == -1:
            raise ValueError("No proof found - missing ':= by'")
            
        # Get everything after the ':= by' line
        proof_lines = lines[proof_start + 1:]
        
        # Remove 2 spaces from start of each line if they exist
        cleaned_lines = [line[2:] if line.startswith("  ") else line for line in proof_lines]
        
        return "\n".join(cleaned_lines)

    @staticmethod
    def _remove_trailing_newlines_and_by(theorem_text: str) -> str:
        """Find the last occurrence of ':= by' and remove it along with everything after it."""
        # Find the last occurrence of ':= by'
        last_by_index = theorem_text.rfind(":= by")
        
        if last_by_index != -1:
            # If found, keep only text before it
            theorem_text = theorem_text[:last_by_index]
        
        # Strip any remaining whitespace
        return theorem_text.strip()
    
    @staticmethod
    def _remove_comment_lines(theorem_text: str) -> str:
        lines = theorem_text.split('\n')
        filtered_lines = []
        for line in lines:
            stripped_line = line.strip()
            # Skip lines starting with '--' (comments)
            if not stripped_line.startswith("--"):
                filtered_lines.append(line)
        clean_theorem_text = '\n'.join(filtered_lines)

        return clean_theorem_text
    
    @staticmethod
    def _split_theorem_text(theorem_text: str) -> Tuple[str, str]:
        """Extract theorem name and rest of text"""
        assert theorem_text.startswith("theorem "), "Invalid theorem text"
        
        # Find where the name ends - either at first parenthesis or at colon
        first_paren = theorem_text.find("(")
        first_colon = theorem_text.find(":")
        
        if first_paren != -1 and first_paren < first_colon:
            # We have hypotheses
            name_end = first_paren
            rest_of_text = theorem_text[first_paren:]
        else:
            # No hypotheses, just conclusion
            name_end = first_colon
            rest_of_text = theorem_text[first_colon:]
            
        name = theorem_text[8:name_end].strip()

        return (name, rest_of_text)
    
    @staticmethod
    def _get_hypothesis(hyp_text: str) -> Hypothesis:
        """Parse hypothesis text into Hypothesis object"""
        # Extract name if present (like h₁)
        name = None
        if ":" in hyp_text:
            name_part = hyp_text[1:hyp_text.find(":")].strip()
            if name_part.startswith("h"):
                name = name_part
    
        hyp = Hypothesis(
            raw_text=hyp_text,
            name=name,
            content=hyp_text[1:-1].strip()
        )
        return hyp

    @staticmethod
    def _parse_theorem(text: str) -> Tuple[List[Hypothesis], str]:
        """Parse theorem text into hypotheses and conclusion."""
        hypotheses = []
        conclusion = ""
        bracket_count = 0
        current_hyp = ""
        
        for i in range(len(text)):
            char = text[i]
            
            if char == "(":
                if bracket_count == 0:
                    # Start of new hypothesis
                    current_hyp = "("
                else:
                    # Nested bracket inside hypothesis
                    current_hyp += char
                bracket_count += 1

            elif char == ")":
                current_hyp += char
                bracket_count -= 1
                
                if bracket_count == 0:
                    # End of current hypothesis
                    hyp = TheoremParser._get_hypothesis(current_hyp)
                    hypotheses.append(hyp)
                    current_hyp = ""
                    
            elif current_hyp:
                # Add characters to the current hypothesis
                current_hyp += char
                
            elif char == ":" and bracket_count == 0:
                # Found the conclusion - split at :=
                remaining_text = text[i+1:].strip()
                # Split conclusion and proof part
                conclusion += remaining_text.split(":=", 1)[0].strip()
                break

        return (hypotheses, conclusion)
    
    @staticmethod
    def parse_have_statements(input_string) -> Dict[str, str]:
        result = {}
        # Split by newlines to get individual statements
        statements = input_string.split('\n')
        
        for statement in statements:
            # Check if this is a "have" statement
            if statement.startswith("have "):
                # Find the position of the first ': ' after 'have '
                colon_pos = statement.find(' : ', 5)  # Start looking after 'have '
                if colon_pos != -1:
                    # Extract name
                    name_part = statement[5:colon_pos].strip()
                    
                    # Use the entire statement as the value
                    result[name_part] = statement
        
        return result
    
    @staticmethod
    def inverse_parse_informal_lemmas(parsed_data):
        # Start with "STEPS:" header
        result = "STEPS:\n"
        
        # Add each lemma with its statement and proof
        for lemma_name, lemma_content in parsed_data['lemmas'].items():
            # Try to get statement/proof or fall back to informal versions
            if 'statement' in lemma_content:
                statement = lemma_content['statement']
            else:
                statement = lemma_content['informal_statement']
                
            if 'proof' in lemma_content:
                proof = lemma_content['proof']
            else:
                proof = lemma_content['informal_proof']
                
            # Format the lemma with its statement and proof
            result += f"{lemma_name}:\n{statement}\nProof:\n{proof}\n\n"
        
        # Add the final proof
        result += f"Final Proof:\n{parsed_data['final_proof']}\n"
        
        return result


def parse_informal_lemmas(informal_lemmas):
    """
    Parse a mathematical proof into its components:
    - final_proof: the concluding proof
    - lemmas: dictionary of lemmas with their statements and proofs
    
    Returns a dictionary with keys 'final_proof' and 'lemmas'
    """
    # Initialize the result dictionary
    result = {
        'final_proof': '',
        'lemmas': {}
    }
    
    # Extract the final proof
    if 'Final Proof:' in informal_lemmas:
        parts = informal_lemmas.split('Final Proof:')
        result['final_proof'] = parts[-1].strip()
    
    # Find all lemma headers using regex for subscript numbers
    lemma_pattern = r'l[₀₁₂₃₄₅₆₇₈₉]+:'
    lemma_matches = list(re.finditer(lemma_pattern, informal_lemmas))
    
    # Process each lemma
    for i, match in enumerate(lemma_matches):
        # Get the lemma name (without the colon)
        lemma_name = match.group()[:-1]
        
        # Determine the start and end positions for this lemma's content
        start_pos = match.end()
        if i < len(lemma_matches) - 1:
            end_pos = lemma_matches[i + 1].start()
        else:
            # If this is the last lemma, end at "Final Proof:" or end of informal_lemmas
            final_proof_pos = informal_lemmas.find('Final Proof:')
            end_pos = final_proof_pos if final_proof_pos != -1 else len(informal_lemmas)
        
        lemma_content = informal_lemmas[start_pos:end_pos].strip()
        
        # Split the lemma content into statement and proof
        if 'Proof:' in lemma_content:
            parts = lemma_content.split('Proof:', 1)
            statement = parts[0].strip()
            proof = parts[1].strip()
            
            # Store in the result dictionary
            result['lemmas'][lemma_name] = {
                'statement': statement,
                'proof': proof
            }
    
    return result


def extract_formal_lemma(text):
    pattern = r"FORMALIZATION:.*?\n\s*(have[^\n]+)"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return ""


def get_subscript(n):
	"""Converts a non-negative integer to its Unicode subscript representation."""
	if not isinstance(n, int) or n < 0:
		raise ValueError("Input must be a non-negative integer")
	subscript_digits = "₀₁₂₃₄₅₆₇₈₉"
	if n == 0:
		return subscript_digits[0]
	return "".join(subscript_digits[int(digit)] for digit in str(n))


def extract_and_rename_have_statements(lean_code: str) -> str:
	"""
	Extracts 'have' statements strictly following the 'have [name] : [statement] := by'
	format, renames them sequentially (l₀, l₁, ...), replaces the proof part
	with just 'by', and removes all leading indentation.

	Ordering follows the rules:
	1. Statements at the same level maintain their original relative order.
	2. Nested statements appear immediately before their parent statement.

	Args:
		lean_code: A string containing the Lean theorem and proof.

	Returns:
		A string containing the modified 'have' statements, separated by newlines,
		with no leading indentation, and ordered according to the rules.
		Returns an empty string if no matching 'have' statements are found.
	"""
	# Regex to capture:
	# Group 1: Indentation (leading whitespace)
	# Group 2: The statement/type part between ':' and ':='
	# Ensures it strictly ends with ':= by' followed by anything else on the line.
	# Allows for an optional original label like 'have h1 :'
	pattern = r"^([ \t]*)\bhave\b(?:\s+\w+)?\s*:\s*(.*?)\s*:=\s*\bby\b.*$" # Changed this line

	# Find all matches and store their indentation and statement
	matches_data = []
	for match in re.finditer(pattern, lean_code, re.MULTILINE):
		indent_len = len(match.group(1))
		statement_part = match.group(2).strip()
		matches_data.append({
			"indent": indent_len,
			"statement": statement_part,
		})

	if not matches_data:
		return "" # No have statements found matching the specific format

	stack = []
	ordered_statements = [] # Will store the statement strings in the correct final order

	for current_match in matches_data:
		current_indent = current_match["indent"]
		current_statement = current_match["statement"]

		# --- Process completed blocks ---
		while stack and stack[-1]["indent"] >= current_indent:
			finished_match = stack.pop()
			ordered_statements.append(finished_match["statement"])

		# --- Push current item ---
		stack.append(current_match)

	# --- Flush remaining stack ---
	while stack:
		finished_match = stack.pop()
		ordered_statements.append(finished_match["statement"])

	# --- Renaming and Formatting ---
	results = []
	for i, stmt in enumerate(ordered_statements):
		new_label = f"l{get_subscript(i)}"
		formatted_line = stmt
		results.append(formatted_line)

	return results


def merge_and_interleave_lemmas(
	proof_attempts_statements: list[list[str]]
) -> list[str]:
	"""
	Merges lists of Lean statement strings (lemmas) from multiple proof attempts
	into a single, ordered list, removing duplicates and attempting to interleave
	them based on local ordering within each attempt.

	Args:
		proof_attempts_statements: A list of lists of strings.
			Each inner list represents a proof attempt and contains
			Lean statement strings in their order of appearance.
			Example: [["x > 0", "y = 1"], ["x > 0", "z = 2", "y = 1"]]

	Returns:
		A single list of unique Lean statement strings, ordered.
	"""
	merged_lemmas_list = []
	seen_lemmas_content = set()
	lemma_to_index_in_merged = {} # Maps lemma string to its index in merged_lemmas_list

	def _update_lemma_to_index_map_from(start_idx_in_merged_list):
		"""Updates lemma_to_index_in_merged for items from start_idx_in_merged_list onwards."""
		for i in range(start_idx_in_merged_list, len(merged_lemmas_list)):
			lemma_to_index_in_merged[merged_lemmas_list[i]] = i

	for current_proof_lemmas in proof_attempts_statements: # Iterate directly over lists of statements
		if not current_proof_lemmas:
			continue

		# Pre-calculate all successors for each lemma IN THE CURRENT PROOF
		successors_map_in_current_proof = collections.defaultdict(set)
		for i, lemma_i in enumerate(current_proof_lemmas):
			for j in range(i + 1, len(current_proof_lemmas)):
				successors_map_in_current_proof[lemma_i].add(current_proof_lemmas[j])

		for current_lemma_idx_in_proof, new_lemma_statement in enumerate(current_proof_lemmas):
			if new_lemma_statement in seen_lemmas_content:
				continue

			# 1. Find latest predecessor of new_lemma_statement (from current_proof_lemmas)
			#	that's already in merged_lemmas_list
			latest_predecessor_merged_idx = -1
			for j in range(current_lemma_idx_in_proof):
				potential_predecessor = current_proof_lemmas[j]
				if potential_predecessor in lemma_to_index_in_merged:
					latest_predecessor_merged_idx = max(
						latest_predecessor_merged_idx,
						lemma_to_index_in_merged[potential_predecessor]
					)

			# 2. Find earliest successor of new_lemma_statement (from current_proof_lemmas)
			#	that's already in merged_lemmas_list
			earliest_successor_merged_idx = len(merged_lemmas_list) # Default: no successor constraint
			for k in range(current_lemma_idx_in_proof + 1, len(current_proof_lemmas)):
				potential_successor = current_proof_lemmas[k]
				if potential_successor in lemma_to_index_in_merged:
					earliest_successor_merged_idx = min(
						earliest_successor_merged_idx,
						lemma_to_index_in_merged[potential_successor]
					)
			
			# 3. Determine initial "wants-to-go-here" index (tentative_idx)
			tentative_idx = 0
			if latest_predecessor_merged_idx != -1:
				tentative_idx = latest_predecessor_merged_idx + 1
			elif earliest_successor_merged_idx != len(merged_lemmas_list):
				tentative_idx = earliest_successor_merged_idx
			else:
				tentative_idx = len(merged_lemmas_list) 

			# Resolve conflict if tentative_idx (based on pred) is after earliest_successor_idx
			if latest_predecessor_merged_idx != -1 and \
			   (latest_predecessor_merged_idx + 1) > earliest_successor_merged_idx:
				tentative_idx = latest_predecessor_merged_idx + 1 # Prioritize predecessor
			
			# 4. Scan forward from tentative_idx to find the actual final_insertion_idx.
			final_insertion_idx = tentative_idx
			scan_upper_bound = earliest_successor_merged_idx # Don't scan past where a successor needs it to be
			
			while final_insertion_idx < scan_upper_bound: 
				if final_insertion_idx >= len(merged_lemmas_list): 
					break
				item_at_slot = merged_lemmas_list[final_insertion_idx]
				if item_at_slot not in successors_map_in_current_proof[new_lemma_statement]:
					final_insertion_idx += 1
				else:
					break 
			
			# Ensure final_insertion_idx itself is not greater than earliest_successor_merged_idx
			if final_insertion_idx > earliest_successor_merged_idx:
				final_insertion_idx = earliest_successor_merged_idx

			# 5. Insert and update
			merged_lemmas_list.insert(final_insertion_idx, new_lemma_statement)
			seen_lemmas_content.add(new_lemma_statement)
			lemma_to_index_in_merged[new_lemma_statement] = final_insertion_idx
			_update_lemma_to_index_map_from(final_insertion_idx + 1)
			
	return merged_lemmas_list


def extract_chosen_lemmas(input_string: str) -> str:
    lines = input_string.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("CHOSEN LEMMAS"):
            return '\n'.join(lines[i+1:])
    return ''


def clean_lean_have_lines(input_str: str) -> str:
    """
    Removes the 'have l<subscript> : ' prefix and ' := by' suffix from lines
    matching that pattern in the input string, specifically targeting subscript indices.

    Args:
        input_str: A multi-line string containing lines potentially in the format
                   'have l<subscript_index> : <statement> := by'.

    Returns:
        A single string containing the extracted <statement> parts from
        matching lines, joined by newline characters.
    """
    cleaned_lines = []
    # Corrected Regex: Using explicit Unicode subscript characters in a character set [].
    # The '+' after the set allows for multi-digit subscripts if they were ever used (e.g., l₁₀).
    # Make sure your editor/environment saves and runs this file as UTF-8.
    pattern_str = r"^\s*have l[₀₁₂₃₄₅₆₇₈₉]+\s*:\s*(.*?)\s*:=\s*by\s*$"
    # re.IGNORECASE affects 'have', 'by'. No UNICODE flag needed with explicit chars.
    pattern = re.compile(pattern_str, re.IGNORECASE)

    # Split the input string into individual lines
    lines = input_str.splitlines()

    for line in lines:
        # Attempt to match the pattern against the current line
        match = pattern.match(line)
        if match:
            # If matched, extract the captured group (the statement)
            # group(1) contains the content within the (.*?) part
            statement = match.group(1)
            cleaned_lines.append(statement)
        # else:
            # Optional: Print lines that don't match for debugging
            # print(f"No match for line: {line}")

    # Join the cleaned lines back into a single string, separated by newlines
    return "\n".join(cleaned_lines)


def get_indentation(line_string: str) -> int:
    """
    Calculates the number of leading whitespace characters in a line.
    Tabs are counted as single characters.
    """
    return len(line_string) - len(line_string.lstrip())


def extract_lemma_proofs(full_proof_text: str) -> dict[str, list[str]]:
    """
    Extracts proof blocks for 'have' lemmas from a given proof text.
    The 'have ... := by ...' line itself is excluded.

    For each unique lemma statement, it stores a list of its corresponding
    proof blocks. Each proof block is normalized so that its first line
    starts with exactly four spaces, and all subsequent lines in that
    block are indented relative to this new base.

    Args:
        full_proof_text: A string containing the entire proof.

    Returns:
        A dictionary where keys are the lemma statement strings (the part
        after 'have name : ' or 'have : ') and values are lists of
        normalized proof block strings.
    """
    proof_map: dict[str, list[str]] = {}
    lines = full_proof_text.splitlines()

    # The target base indentation for the output proof blocks.
    TARGET_BASE_INDENT = 4 # Changed from 2 to 4

    lemma_line_identifier_pattern = re.compile(
        r"^\s*have\s+(?P<statement_content>.*?)\s*:=\s*by(?P<inline_tactic>.*)$"
    )
    
    name_statement_pattern = re.compile(r"^(?P<name>[^:]+?)\s*:\s*(?P<actual_statement>.*)$")

    for i, line_text in enumerate(lines):
        line_match = lemma_line_identifier_pattern.match(line_text)
        
        if line_match:
            content_between_have_and_by = line_match.group("statement_content").strip()
            inline_tactic_part = line_match.group("inline_tactic").strip()
            
            statement_text = ""
            ns_match = name_statement_pattern.match(content_between_have_and_by)
            if ns_match:
                statement_text = ns_match.group("actual_statement").strip()
            elif content_between_have_and_by.startswith(":") and len(content_between_have_and_by) > 1:
                statement_text = content_between_have_and_by[1:].strip()
            elif not content_between_have_and_by.startswith(":"):
                statement_text = content_between_have_and_by
            else:
                statement_text = ""

            original_have_line_indent = get_indentation(line_text)
            actual_proof_block_lines_raw = []

            for j in range(i + 1, len(lines)):
                next_line_text = lines[j]
                next_line_indent = get_indentation(next_line_text)
                
                if next_line_indent > original_have_line_indent:
                    actual_proof_block_lines_raw.append(next_line_text)
                else:
                    break
            
            normalized_proof_lines = []

            if actual_proof_block_lines_raw:
                first_content_line_index = -1
                for k, line in enumerate(actual_proof_block_lines_raw):
                    if line.strip():
                        first_content_line_index = k
                        break
                
                if first_content_line_index != -1:
                    first_proof_line_original_indent = get_indentation(actual_proof_block_lines_raw[first_content_line_index])
                    
                    for k, raw_line in enumerate(actual_proof_block_lines_raw):
                        stripped_line_content = raw_line.lstrip()
                        
                        # Handle empty lines before the first content line consistently
                        if not stripped_line_content and (k < first_content_line_index or not actual_proof_block_lines_raw[0].strip()):
                             # If it's an empty line before first content, or the whole block starts empty,
                             # just give it the base indent.
                            normalized_proof_lines.append((" " * TARGET_BASE_INDENT) + stripped_line_content)
                            continue

                        original_line_indent = get_indentation(raw_line)
                        relative_indent = original_line_indent - first_proof_line_original_indent
                        new_indent_count = max(0, TARGET_BASE_INDENT + relative_indent)
                        normalized_proof_lines.append((" " * new_indent_count) + stripped_line_content)
                # If actual_proof_block_lines_raw only contains empty/whitespace lines before any content
                # (e.g. block is just "  \n    \n"), this logic will make them all TARGET_BASE_INDENT
                elif actual_proof_block_lines_raw: # All lines are whitespace
                    for raw_line in actual_proof_block_lines_raw:
                         normalized_proof_lines.append((" " * TARGET_BASE_INDENT) + raw_line.lstrip())


            elif inline_tactic_part:
                normalized_proof_lines.append((" " * TARGET_BASE_INDENT) + inline_tactic_part)
            

            if normalized_proof_lines:
                normalized_proof_string = "\n".join(normalized_proof_lines)
                if statement_text not in proof_map:
                    proof_map[statement_text] = []
                proof_map[statement_text].append(normalized_proof_string)
            
    return proof_map


def get_statement_from_have_line(have_line_string: str) -> str | None:
    """
    Extracts the statement from a single 'have ... := by ...' line.
    Returns the statement string, or None if parsing fails or not a have line.
    """
    lemma_line_identifier_pattern = re.compile(
        r"^\s*have\s+(.*?)\s*:=\s*by(?:.*)?$" # Allows tactics on the same line after 'by'
    )
    name_statement_pattern = re.compile(r"^(?P<name>[^:]+?)\s*:\s*(?P<actual_statement>.*)$")

    line_match = lemma_line_identifier_pattern.match(have_line_string)
    if line_match:
        content_between_have_and_by = line_match.group(1).strip()
        statement_text = ""
        ns_match = name_statement_pattern.match(content_between_have_and_by)
        if ns_match:
            statement_text = ns_match.group("actual_statement").strip()
        elif content_between_have_and_by.startswith(":"):
            statement_text = content_between_have_and_by[1:].strip()
        else:
            statement_text = content_between_have_and_by
        return statement_text
    return None


def map_target_lemmas_to_failed_proofs(
    target_lemma_lines: list[str], 
    failed_attempt_texts: list[str]
) -> dict[str, list[str]]:
    """
    Maps target lemma lines to a list of proofs for their statements found in failed attempts.

    Args:
        target_lemma_lines: A list of strings, each being a full 'have ... := by' line
                            representing a target lemma.
        failed_attempt_texts: A list of strings, each string being the full text of a
                              failed proof attempt.

    Returns:
        A dictionary where keys are the original target lemma line strings and
        values are lists of normalized proof block strings found in any of the
        failed attempts that match the target lemma's statement.
    """
    all_proofs_from_failures_by_statement: dict[str, list[str]] = {}

    # 1. Process all failed attempts and aggregate proofs by statement
    for attempt_text in failed_attempt_texts:
        lemmas_in_attempt = extract_lemma_proofs(attempt_text)
        for statement, proof_blocks in lemmas_in_attempt.items():
            if statement not in all_proofs_from_failures_by_statement:
                all_proofs_from_failures_by_statement[statement] = []
            all_proofs_from_failures_by_statement[statement].extend(proof_blocks)

    output_map: dict[str, list[str]] = {}

    # 2. Process target lemmas and map them to the aggregated proofs
    for original_target_lemma_line in target_lemma_lines:
        target_statement = get_statement_from_have_line(original_target_lemma_line)

        if target_statement is not None:
            # Get proofs for this statement, or an empty list if statement not found in failures
            found_proofs = all_proofs_from_failures_by_statement.get(target_statement, [])
            output_map[original_target_lemma_line] = found_proofs
        else:
            # If the target_lemma_line itself isn't a valid 'have' line,
            # or statement couldn't be extracted.
            # Assign an empty list, or you could log a warning/error.
            output_map[original_target_lemma_line] = [] 
            print(f"Warning: Could not parse statement from target line: '{original_target_lemma_line}'")


    return output_map


def remove_two_leading_spaces_from_lines(text: str) -> str:
    """
    Removes the first two leading spaces from each line of the input string,
    if a line starts with at least two spaces.

    - If a line has two or more leading spaces, the first two are removed.
      Example: "  foo" becomes "foo", "   bar" becomes " bar".
    - If a line has fewer than two leading spaces (i.e., zero or one),
      it remains unchanged.
      Example: " foo" remains " foo", "baz" remains "baz".
    - Empty lines remain empty.
    - A line consisting of exactly "  " (two spaces) becomes an empty line "".
    - A line consisting of " " (one space) remains " ".

    Args:
        text: The input string, possibly multi-line.

    Returns:
        A new string with the first two leading spaces removed from applicable lines.
    """
    if not text:
        return ""

    lines = text.splitlines(keepends=False) # Process line by line
    processed_lines = []
    
    prefix_to_remove = "  " # The two spaces we're looking for

    for line in lines:
        if line.startswith(prefix_to_remove):
            # If the line starts with two spaces, remove them
            processed_lines.append(line[len(prefix_to_remove):])
        else:
            # Otherwise, keep the line as is
            processed_lines.append(line)
    
    # Join the processed lines back together with newlines
    return "\n".join(processed_lines)


def to_subscript(n: int) -> str:
	"""Converts a non-negative integer to its Unicode subscript string representation.
	Example: 0 -> "₀", 12 -> "₁₂"
	"""
	if not isinstance(n, int):
		raise TypeError("Input must be an integer.")
	if n < 0:
		# This case might need specific handling if negative subscripts are possible,
		# but typically lemma indices are non-negative.
		raise ValueError("Subscript conversion is for non-negative integers.")
		
	s = str(n)
	subscript_map = {
		'0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
		'5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'
	}
	# Convert each digit in the number to its subscript equivalent
	return "".join(subscript_map[digit] for digit in s)