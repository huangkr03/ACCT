import os
import tempfile

def check(checker, input, correct_output, solution_output):
    with tempfile.TemporaryDirectory() as tmpdir:
        checker_file  = os.path.join(tmpdir, 'checker.py')
        input_file = os.path.join(tmpdir, 'input.txt')
        correct_output_file = os.path.join(tmpdir, 'correct_output.txt')
        solution_output_file = os.path.join(tmpdir, 'solution_output.txt')
        
        with open(checker_file, 'w') as f:
            f.write(checker)
        
        with open(input_file, 'w') as f:
            f.write(input)
        
        with open(correct_output_file, 'w') as f:
            f.write(correct_output)
            
        with open(solution_output_file, 'w') as f:
            f.write(solution_output)
        os.system(f'python {checker_file} {input_file} {correct_output_file} {solution_output_file}')