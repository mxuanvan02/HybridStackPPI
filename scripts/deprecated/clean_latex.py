import re

def remove_color_blocks(text, colorCommand):
    result = ""
    idx = 0
    while idx < len(text):
        pos = text.find(colorCommand, idx)
        if pos == -1:
            result += text[idx:]
            break
        result += text[idx:pos]
        
        # Now find the matching closing brace
        brace_count = 1
        i = pos + len(colorCommand)
        while i < len(text) and brace_count > 0:
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
            i += 1
        
        if brace_count == 0:
            # We skip adding text[pos:i] to result, effectively deleting the block
            idx = i
        else:
            # Unmatched brace, just skip the command for safety
            result += colorCommand
            idx = pos + len(colorCommand)
    return result

def unwrap_color_blocks(text, colorCommand):
    result = ""
    idx = 0
    while idx < len(text):
        pos = text.find(colorCommand, idx)
        if pos == -1:
            result += text[idx:]
            break
        result += text[idx:pos]
        
        # Now find the matching closing brace
        brace_count = 1
        i = pos + len(colorCommand)
        inner_start = i
        while i < len(text) and brace_count > 0:
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
            i += 1
        
        if brace_count == 0:
            # Add the inner content WITHOUT the wrapper
            result += text[inner_start:i-1]
            idx = i
        else:
            result += colorCommand
            idx = pos + len(colorCommand)
    return result

with open("IEEE_Access/ACCESS_latex_template_20240429/HybridStackingPPI.tex", "r") as f:
    content = f.read()

# 1. Remove red blocks
content = remove_color_blocks(content, r"{\color{red}")

# 2. Unwrap blue blocks
content = unwrap_color_blocks(content, r"{\color{blue}")

with open("IEEE_Access/ACCESS_latex_template_20240429/HybridStackingPPI.tex", "w") as f:
    f.write(content)

print("Red blocks removed and blue blocks unwrapped.")
