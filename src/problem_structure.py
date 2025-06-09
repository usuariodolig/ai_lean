from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from theorem_manager import TheoremManager, ProofStructure


@dataclass
class Attempt:
	"""Represents an attempt to prove a theorem or lemma."""
	success: bool
	proof: str
	message: List
	generation_time: float
	verification_time: float
		
	def to_dict(self) -> Dict[str, Any]:
		"""Convert the Attempt to a dictionary."""
		return {
			"success": self.success,
			"proof": self.proof,
			"message": self.message,
			"generation_time": self.generation_time,
			"verification_time": self.verification_time
		}
		
	@staticmethod
	def from_dict(attempt_dict: Dict) -> Attempt:
		return Attempt(
			success=attempt_dict['success'],
			proof=attempt_dict['proof'],
			message=attempt_dict['message'],
			generation_time=attempt_dict['generation_time'],
			verification_time=attempt_dict['verification_time']
		)


class BreakdownAttempt:
	# Lemmas should already be formalized
	def __init__(self, formal_statement: str, nl_proof: str, proof_structure: ProofStructure):
		self.nl_proof = nl_proof
		self.proof_structure = proof_structure
		self.theorem_manager = TheoremManager(formal_statement, proof_structure)
		self.initial_breakdown_attempts: List[Attempt] = []
		self.final_breakdown_attempts: List[Attempt] = []
		self.lemmas: Dict[str, TheoremProcessor] = dict()
		self.solution: Optional[str] = None
		
	def add_initial_breakdown_attempt(self, attempt: Attempt) -> None:
		"""Add an initial theorem attempt."""
		self.initial_breakdown_attempts.append(attempt)
		if attempt.success:
			self.theorem_manager.add_formal_proof(attempt.proof)
		
	def add_final_breakdown_attempt(self, attempt: Attempt) -> None:
		"""Add a final theorem attempt."""
		self.final_breakdown_attempts.append(attempt)
		
		if attempt.success:
			self.theorem_manager.add_formal_proof(attempt.proof)
			self.solution = self.theorem_manager.get_theorem_with_proofs()
		
	def has_successful_initial_breakdown_attempt(self) -> bool:
		"""Check if there's a successful initial theorem attempt."""
		return any(attempt.success for attempt in self.initial_breakdown_attempts)
		
	def get_successful_initial_breakdown_attempt(self) -> str:
		if self.has_successful_initial_breakdown_attempt():
			for attempt in self.initial_breakdown_attempts:
				if attempt.success:
					return attempt.proof
		else:
			return ""
		
	def check_if_can_be_proven_from_proven_lemmas_and_add(self, proof: str, unused_lemmas: List[str]) -> None:
		if not proof:
			return
		unproven_lemmas = self._get_unproven_lemmas()
		for lemma in unproven_lemmas:
			if lemma not in unused_lemmas:
				return
		# If all unproven lemmas are unused in the proof, add proof to solution
		self.theorem_manager.add_formal_proof(proof)
		self.solution = self.theorem_manager.get_theorem_with_proofs()
		
	def add_lemma(self, lemma_name, lemma: TheoremProcessor) -> None:
		self.lemmas[lemma_name] = lemma
		
	def has_solution(self) -> bool:
		"""Check if the problem has a solution."""
		return self.solution is not None
		
	def get_solution(self) -> str:
		return self.solution
		
	def add_lemma_proof(self, lemma_name: str, formal_proof: str) -> None:
		self.theorem_manager.add_lemma_formal_proof_from_name(lemma_name, formal_proof)

	def update_lemma_statements(self) -> None:
		for i, lemma in enumerate(self.lemmas.values()):
			lemma.formal_statement = self.theorem_manager.get_lemma_as_theorem(i)

	def _proved_all_lemmas(self) -> bool:
		if not self.lemmas:
			return False
		
		for lemma in self.lemmas.values():
			if not lemma.solution:
				return False
		return True
		
	def _get_unproven_lemmas(self) -> List[str]:
		unproven_lemmas = []
		for lemma_name, lemma in self.lemmas.items():
			if not lemma.has_solution():
				unproven_lemmas.append(lemma_name)
		return unproven_lemmas

	def check_if_has_solution_and_add_it(self) -> None:
		if self._proved_all_lemmas() and self.has_successful_initial_breakdown_attempt():
			for attempt in self.initial_breakdown_attempts:
				if attempt.success:
					self.theorem_manager.add_formal_proof(attempt.proof)
					self.solution = self.theorem_manager.get_theorem_with_proofs()

	def count_attempts(self) -> int:
		count = 0
		count += len(self.initial_breakdown_attempts)
		count += len(self.final_breakdown_attempts)
		for lemma in self.lemmas.values():
			count += lemma.count_attempts()
		return count

	def to_dict(self) -> Dict[str, Any]:
		return {
			"nl_proof": self.nl_proof,
			"proof_structure": self.proof_structure.to_dict(),
			"initial_breakdown_attempts": [attempt.to_dict() for attempt in self.initial_breakdown_attempts],
			"final_breakdown_attempts": [attempt.to_dict() for attempt in self.final_breakdown_attempts],
			"lemmas": {name: lemma.to_dict() for name, lemma in self.lemmas.items()}
		}
		
	@staticmethod
	def from_dict(breakdown_attempt_dict: Dict, formal_statement: str) -> BreakdownAttempt:
		proof_structure = ProofStructure.from_dict(breakdown_attempt_dict['proof_structure'])
		breakdown_attempt = BreakdownAttempt(
			formal_statement=formal_statement,
			nl_proof=breakdown_attempt_dict['nl_proof'],
			proof_structure=proof_structure
		)
		# We need to fix the TheoremManager, other than that it's all easy
		breakdown_attempt.theorem_manager = TheoremManager(formal_statement, proof_structure)
		# MORE IS NEEDED, but what? Maybe nothing but we'll have to check

		# Assume we fixed the TheoremManager
		breakdown_attempt.proof_structure = proof_structure
		breakdown_attempt.initial_breakdown_attempts = [Attempt.from_dict(attempt) for attempt in breakdown_attempt_dict['initial_breakdown_attempts']]
		breakdown_attempt.final_breakdown_attempts = [Attempt.from_dict(attempt) for attempt in breakdown_attempt_dict['final_breakdown_attempts']]
		breakdown_attempt.lemmas = {name: TheoremProcessor.from_dict(lemma) for name, lemma in breakdown_attempt_dict['lemmas'].items()}
		# This all looks good. Now what's left?
		# Adding the lemmas to the TheoremManager
		for lemma_name, lemma in breakdown_attempt_dict['lemmas'].items():
			breakdown_attempt.theorem_manager.add_lemma_formal_proof_from_name(lemma_name, lemma['solution'])

		# These are the only two ways a BreakdownAttempt can gain a solution
		if breakdown_attempt.final_breakdown_attempts and breakdown_attempt.final_breakdown_attempts[-1].success:
			breakdown_attempt.solution = breakdown_attempt.final_breakdown_attempts[-1].proof
			breakdown_attempt.theorem_manager.add_formal_proof(breakdown_attempt.solution) # idk if this is right
		elif breakdown_attempt._proved_all_lemmas() and breakdown_attempt.initial_breakdown_attempts:
			breakdown_attempt.solution = breakdown_attempt.initial_breakdown_attempts[-1].proof
			breakdown_attempt.theorem_manager.add_formal_proof(breakdown_attempt.solution) # idk if this is right

		return breakdown_attempt


class TheoremProcessor:
	def __init__(self, formal_statement: str, informal_statement: str, informal_proof: str):
		self.formal_statement = formal_statement
		self.informal_statement = informal_statement
		self.informal_proof = informal_proof
		self.initial_attempts: List[Attempt] = []
		self.breakdown_attempts: List[BreakdownAttempt] = []
		self.solution: Optional[str] = None
		
	def add_initial_attempt(self, attempt: Attempt) -> None:
		self.initial_attempts.append(attempt)
		
		if attempt.success:
			self.solution = attempt.proof
		
	def add_breakdown_attempt(self, nl_proof: str, proof_structure: ProofStructure) -> BreakdownAttempt:
		self.breakdown_attempts.append(BreakdownAttempt(self.formal_statement, nl_proof, proof_structure))
		return self.breakdown_attempts[-1]
		
	def has_solution(self) -> bool:
		return self.solution is not None
		
	def add_solution(self, solution: str) -> None:
		self.solution = solution

	def count_attempts(self) -> int:
		count = 0
		count += len(self.initial_attempts)
		for breakdown in self.breakdown_attempts:
			count += breakdown.count_attempts()
		return count

	def to_dict(self) -> Dict[str, Any]:
		return {
			"formal_statement": self.formal_statement,
			"informal_statement": self.informal_statement,
			"informal_proof": self.informal_proof,
			"initial_attempts": [attempt.to_dict() for attempt in self.initial_attempts],
			"breakdown_attempts": [breakdown.to_dict() for breakdown in self.breakdown_attempts],
			"solution": self.solution
		}
		
	@staticmethod
	def from_dict(theorem_processor_dict: Dict) -> TheoremProcessor:
		# This is working (based on only 1 example LMAO)
		theorem_processor = TheoremProcessor(
			formal_statement=theorem_processor_dict['formal_statement'],
			informal_statement=theorem_processor_dict['informal_statement'],
			informal_proof=theorem_processor_dict['informal_proof']
		)
		theorem_processor.initial_attempts = [Attempt.from_dict(attempt) for attempt in theorem_processor_dict['initial_attempts']]
		theorem_processor.breakdown_attempts = [BreakdownAttempt.from_dict(breakdown, theorem_processor.formal_statement) for breakdown in theorem_processor_dict['breakdown_attempts']]
		theorem_processor.solution = theorem_processor_dict['solution']

		return theorem_processor