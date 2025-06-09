from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict
import re

from .formatting import Hypothesis, TheoremParser


@dataclass
class LemmaEntry:
    informal_statement: str
    informal_proof: str
    formal_statement: Optional[str] = None

    def to_dict(self):
        lemma_entry_dict = {
            'informal_statement': self.informal_statement,
            'informal_proof': self.informal_proof,
            'formal_statement': self.formal_statement
        }
        return lemma_entry_dict


@dataclass
class Lemma(Hypothesis):
    informal_statement: str
    informal_proof: Optional[str] = None
    formal_proof: Optional[str] = None


class ProofStructure:
    def __init__(self, informal_lemmas):
        lemmas_dict = self._parse_informal_lemmas(informal_lemmas)
        self.final_proof = lemmas_dict['final_proof']
        self.lemmas = dict()
        for lemma_name in lemmas_dict['lemmas']:
            lemma_entry = LemmaEntry(
                informal_statement = lemmas_dict['lemmas'][lemma_name]['statement'],
                informal_proof = lemmas_dict['lemmas'][lemma_name]['proof']
            )
            self.lemmas[lemma_name] = lemma_entry

    def _parse_informal_lemmas(self, informal_lemmas: str):
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
    
    def add_formal_statement(self, lemma_name: str, formal_statement: str) -> None:
        lemma = self.lemmas[lemma_name]
        lemma.formal_statement = formal_statement

    def to_dict(self):
        proof_structure_dict = {
            'final_proof': self.final_proof,
            'lemmas': dict()
        }
        for lemma_name in self.lemmas:
            lemma = self.lemmas[lemma_name].to_dict()
            proof_structure_dict['lemmas'][lemma_name] = lemma

        return proof_structure_dict
    
    @staticmethod
    def from_dict(proof_structure_dict) -> ProofStructure:
        informal_lemmas = TheoremParser.inverse_parse_informal_lemmas(proof_structure_dict)
        proof_structure = ProofStructure(informal_lemmas)
        for lemma_name, lemma in proof_structure_dict['lemmas'].items():
            proof_structure.add_formal_statement(lemma_name, lemma['formal_statement'])
        return proof_structure


class LemmaManager:
    def __init__(self, lemmas: Optional[Dict[str, LemmaEntry]] = None):
        self.lemmas: List[Lemma] = []
        if lemmas:
            for lemma in lemmas.values():
                self.add_lemma(lemma)

    def add_lemma(self, lemma_entry: LemmaEntry) -> None:
        if not lemma_entry.formal_statement:
            raise ValueError("LemmaEntry doesn't have a formal statement")
        
        formal_statement = lemma_entry.formal_statement
        informal_statement = lemma_entry.informal_statement
        informal_proof = lemma_entry.informal_proof
        # Remove leading/trailing whitespace
        formal_statement = formal_statement.strip()
        
        # Basic validation
        if not formal_statement.startswith("have "):
            raise ValueError("Statement must start with 'have'")
            
        # Remove "have " prefix
        formal_statement = formal_statement[5:]
        
        # Find the name part (h₁, h₂, etc.)
        colon_idx = formal_statement.find(" : ")
        if colon_idx == -1:
            raise ValueError("Invalid have-statement format: missing ' : '")
            
        name = formal_statement[:colon_idx].strip()
            
        # Extract the actual formal_statement content
        formal_statement = formal_statement[colon_idx + 2:]
        
        # Remove := by ... if present
        if ":= by" in formal_statement:
            formal_statement = formal_statement.split(":= by")[0].strip()
            
        # Create and add the lemma
        lemma = Lemma(
            raw_text=f"({name} : {formal_statement})",
            name=name,
            content=formal_statement,
            informal_statement=informal_statement,
            informal_proof=informal_proof
        )
        self.lemmas.append(lemma)

    def add_formal_proof(self, i: int, formal_proof: str) -> None:
        """
        Add a proof for a lemma.
        
        Args:
            i: Index of the lemma to add proof for
            proof: Proof text
        """
        if not 0 <= i < len(self.lemmas):
            raise ValueError(f"Invalid lemma index: {i}")
            
        self.lemmas[i].formal_proof = TheoremParser.extract_proof(formal_proof)

    def add_formal_proof_from_name(self, lemma_name: str, formal_proof: str) -> None:
        for i, lemma in enumerate(self.lemmas):
            if lemma.name == lemma_name:
                self.lemmas[i].formal_proof = TheoremParser.extract_proof(formal_proof)


class TheoremManager:
    def __init__(self, formal_statement: str, proof_structure: ProofStructure, lemmas_contain_proof=True): # The proof structure should already contain the formalized lemmas
        self.name, self.hypotheses, self.conclusion = TheoremParser.parse(formal_statement)
        self.proof_structure = proof_structure
        self.informal_proof = proof_structure.final_proof
        if lemmas_contain_proof:
            self.lemma_manager = LemmaManager(proof_structure.lemmas)
        else:
            self.lemma_manager = LemmaManager()
        self.lemmas = self.lemma_manager.lemmas
        self.formal_proof = ""
    
    def get_theorem_statement(self, include_lemmas_as_hypothesis=False, skip_unproven_lemmas=True) -> str:
        """Get properly formatted theorem statement."""
        lines = [f"theorem {self.name}"]
        
        # Add all hypotheses
        for hyp in self.hypotheses:
            lines.append(f"  {hyp.raw_text}")
            
        # Add lemmas if requested
        if include_lemmas_as_hypothesis:
            for lemma in self.lemmas:
                if lemma.formal_proof or not skip_unproven_lemmas:
                    lines.append(f"  {lemma.raw_text}")
            
        # Add conclusion
        lines.append(f"  : {self.conclusion} := by")
        
        return "\n".join(lines)
        
    def get_lemma_as_theorem(self, lemma_index: int, include_informal_proof=False, skip_unproven_lemmas=True) -> str:
        """
        Format a specific lemma as a theorem, including all original hypotheses
        and previous lemmas as hypotheses.
        
        Args:
            lemma_index: Index of the lemma to use as conclusion
            
        Returns:
            Formatted theorem text with the lemma as conclusion
        """
        if not 0 <= lemma_index < len(self.lemmas):
            raise ValueError(f"Invalid lemma index: {lemma_index}")
            
        # Start with theorem name and original hypotheses
        lines = [f"theorem {self.name}_lemma_{lemma_index}"]
        
        # Add original hypotheses
        for hyp in self.hypotheses:
            lines.append(f"  {hyp.raw_text}")
            
        # Add all lemmas before the target lemma as hypotheses
        for i in range(lemma_index):
            if skip_unproven_lemmas:
                if self.lemmas[i].formal_proof: # Skip unproven lemmas
                    lines.append(f"  {self.lemmas[i].raw_text}")
            else:
                lines.append(f"  {self.lemmas[i].raw_text}")
            
        # Add the target lemma's content as the conclusion
        target_lemma = self.lemmas[lemma_index]
        lines.append(f"  : {target_lemma.content} := by")

        if include_informal_proof:
            # Add proof if it exists
            if target_lemma.informal_proof:
                lines.append('  /-')
                informal_proof_lines = target_lemma.informal_proof.splitlines()
                for proof_line in informal_proof_lines:
                    proof_line = '  ' + proof_line
                    lines.append(proof_line)
                lines.append('  -/')
        
        return "\n".join(lines)
    
    def get_theorem_with_proofs(self) -> str:
        # Start with the theorem statement
        lines = [f"theorem {self.name}"]
        
        # Add hypotheses
        for hyp in self.hypotheses:
            lines.append(f"  {hyp.raw_text}")
            
        # Add conclusion
        lines.append(f"  : {self.conclusion} := by")
        
        # Add each lemma with its proof
        for lemma in self.lemmas:
            if lemma.formal_proof:
                lines.append(f"  have {lemma.name} : {lemma.content} := by")
                # Split proof into lines and add proper indentation
                proof_lines = lemma.formal_proof.split('\n')
                for line in proof_lines:
                    if line.strip():  # If line is not empty
                        lines.append(f"    {line}")
        
        # Add the final theorem proof if it exists
        if self.formal_proof:
            # Split final proof into lines and add proper indentation
            proof_lines = self.formal_proof.split('\n')
            for line in proof_lines:
                if line.strip():  # If line is not empty
                    lines.append(f"  {line}")
        else:
            lines.append("  sorry")
        
        return "\n".join(lines)
    
    def __len__(self):
        return len(self.lemmas)
    
    def add_lemma(self, lemma_entry: LemmaEntry) -> None:
        self.lemma_manager.add_lemma(lemma_entry)

    def add_formal_proof(self, formal_proof: str) -> None:
        self.formal_proof = TheoremParser.extract_proof(formal_proof)

    def add_lemma_formal_proof(self, i: int, formal_proof: str) -> None:
        self.lemma_manager.add_formal_proof(i, formal_proof)

    def add_lemma_formal_proof_from_name(self, lemma_name: str, formal_proof: str) -> None:
        self.lemma_manager.add_formal_proof_from_name(lemma_name, formal_proof)