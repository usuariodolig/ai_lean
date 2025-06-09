import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src import theorem_proving
from src.problem_structure import TheoremProcessor


def count_lemmas(processor: TheoremProcessor) -> int:
	if not processor.breakdown_attempts:
		return 0
	lemmas = processor.breakdown_attempts[0].lemmas
	return len(lemmas)

BATCH_SIZE = 32

if __name__ == '__main__':
	# Model initialization
	model_path = "/workspace/deepseek/models--deepseek-ai--DeepSeek-Prover-V1.5-RL/snapshots/40a760138c68038d9869a436077c16b654b1cc72"
	
	tokenizer = AutoTokenizer.from_pretrained(
		model_path,
		local_files_only=True  # Only use local files
	)

	model = AutoModelForCausalLM.from_pretrained(
		model_path,
		torch_dtype=torch.float16,
		local_files_only=True  # Only use local files
	).cuda()

	model.eval()

	with open('proofnet.json', 'r', encoding='utf-8') as f:
		proofnet_test = json.load(f)

	proofnet_lookup_by_statement = {problem['formal_statement']: problem for problem in proofnet_test}
	print(len(proofnet_lookup_by_statement))


	with open('tests/breakdown_attempts_nl_proof.json', 'r', encoding='utf-8') as f:
		processors_nl_proof_dict = json.load(f)
	
	with open('tests/breakdown_attempts_vanilla.json', 'r', encoding='utf-8') as f:
		processors_vanilla_dict = json.load(f)
	
	processors_nl_proof = dict()
	for processor_dict in processors_nl_proof_dict:
		theorem_processor = TheoremProcessor.from_dict(processor_dict)
		problem = proofnet_lookup_by_statement[theorem_processor.formal_statement]
		processors_nl_proof[problem['name']] = theorem_processor

	processors_vanilla = dict()
	for processor_dict in processors_vanilla_dict:
		theorem_processor = TheoremProcessor.from_dict(processor_dict)
		problem = proofnet_lookup_by_statement[theorem_processor.formal_statement]
		processors_vanilla[problem['name']] = theorem_processor
	
	i = 0
	while True:
		i += 1
		stop = True
		print(f"============================== {i} ==============================")
		# Part with nl proofs
		for j, theorem_processor in enumerate(processors_nl_proof.values()):
			if theorem_processor.has_solution():
				continue
			elif theorem_processor.count_attempts() >= 16 + 112 * (1 + count_lemmas(theorem_processor)):
				continue
			
			stop = False
			print(f"with nl_proof: {j}")
			problem = proofnet_lookup_by_statement[theorem_processor.formal_statement]
			header = problem['header']
			name = problem['name']
			
			try: 
				theorem_proving.attempt_more_proofs_batch(
					theorem_processor, model, tokenizer, None,
					n_initial_breakdown_attempts=BATCH_SIZE,
					n_attempts_per_lemma=BATCH_SIZE,
					use_llm_cot=True,
					cot_proof=theorem_processor.informal_proof,
					header=header
				)

				with open(f'tests/nl_proof/{name}.json', 'w', encoding='utf-8') as file:
					json.dump({problem['name'] : theorem_processor.to_dict()}, file, indent=4)
			except Exception as e:
				print("ERROR: ", e)
		
		# Part with vanilla cot
		for j, theorem_processor in enumerate(processors_vanilla.values()):
			if theorem_processor.has_solution():
				continue
			elif theorem_processor.count_attempts() >= 16 + 112 * (1 + count_lemmas(theorem_processor)): 
				continue
			
			stop = False
			print(f"vanilla cot  : {j}")
			problem = proofnet_lookup_by_statement[theorem_processor.formal_statement]
			header = problem['header']
			name = problem['name']
			
			try:
				theorem_proving.attempt_more_proofs_batch(
					theorem_processor, model, tokenizer, None,
					n_initial_breakdown_attempts=BATCH_SIZE,
					n_attempts_per_lemma=BATCH_SIZE,
					use_llm_cot=False,
					cot_proof="",
					header=header
				)

				with open(f'tests/vanilla/{name}.json', 'w', encoding='utf-8') as file:
					json.dump({problem['name'] : theorem_processor.to_dict()}, file, indent=4)
			except Exception as e:
				print("ERROR: ", e)
		
		if stop:
			break