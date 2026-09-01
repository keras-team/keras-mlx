"""Check that every line of `excluded_tests.txt` means what it looks like.

keras selects the tests to skip for this backend with a plain substring
test, `if skipped_test in item.nodeid`, so the file has no anchors and no
comment syntax. A line that reads like one test id can quietly gate a whole
class, and a line that another line already covers is inert. Neither is
visible when reading the file, so check for both here.
"""

import os
import sys

EXCLUSIONS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "excluded_tests.txt",
)


def check(lines):
    """Find the lines of the exclusion file that do not mean what they say.

    Args:
        lines: the lines of the file, without their newlines.

    Returns:
        A list of `(line_number, message)` tuples, empty when the file is
        clean. Line numbers are 1 based, to match an editor.
    """
    problems = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            problems.append(
                (
                    number,
                    "Blank line. keras drops these, so it gates nothing and "
                    "only makes the file harder to scan.",
                )
            )
            continue
        if line != line.strip():
            problems.append(
                (
                    number,
                    "Leading or trailing whitespace. keras strips it before "
                    "matching, so the pattern is not what it reads as. "
                    f"Received: line={line!r}",
                )
            )
            continue
        if line.startswith("#"):
            problems.append(
                (
                    number,
                    "The file has no comment syntax. This is a live pattern "
                    f"that matches no test id. Received: line={line!r}",
                )
            )
            continue
        if "::" not in line:
            problems.append(
                (
                    number,
                    "No `::`, so this gates every test of the class rather "
                    "than one test. Name the tests instead. "
                    f"Received: line={line!r}",
                )
            )

    stripped = [line.strip() for line in lines]
    seen = {}
    for number, line in enumerate(stripped, start=1):
        if not line:
            continue
        if line in seen:
            problems.append(
                (number, f"Duplicate of line {seen[line]}. Received: {line!r}")
            )
        else:
            seen[line] = number

    # A shorter pattern that a longer one contains already skips everything
    # the longer one does, which makes the longer one inert. Deleting the
    # longer one then reads like a change and is not one.
    for number, line in enumerate(stripped, start=1):
        if not line:
            continue
        for other in stripped:
            if other and other != line and other in line:
                problems.append(
                    (
                        number,
                        f"Already covered by {other!r}, which this line "
                        "contains, so this line skips nothing on its own.",
                    )
                )
                break

    # Report only the first pair out of order. Fixing it changes where the
    # next one is, so listing them all would be noise.
    listed = [(number, line) for line, number in seen.items()]
    for (_, first), (number, second) in zip(listed, listed[1:]):
        if second < first:
            problems.append(
                (
                    number,
                    f"Out of order, {second!r} sorts before {first!r}. "
                    "Keeping the file sorted is what makes a near duplicate "
                    "visible in review.",
                )
            )
            break

    return sorted(problems)


def main():
    with open(EXCLUSIONS_PATH) as file:
        lines = file.read().splitlines()

    name = os.path.basename(EXCLUSIONS_PATH)
    problems = check(lines)
    for number, message in problems:
        print(f"{name}:{number}: {message}")

    if problems:
        print(f"\n{len(problems)} problems in {name}.")
        return 1

    print(f"{name} is clean, {len(lines)} patterns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
