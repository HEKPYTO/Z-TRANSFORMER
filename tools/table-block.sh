#!/bin/sh
# Fails unless the tables a tool prints are the tables a README quotes.
#
#   sh tools/table-block.sh <tool-output> <readme> <block> <floor> <header>...
#
# `zig build scale-profile` prints three tables and `src/README.md` quotes them
# between `<!-- scale-profile:begin -->` and `<!-- scale-profile:end -->`. This
# is the comparison behind that gate, and it is a script in the tree rather than
# a block of shell inside `build.zig` for one reason: `tools/table-block-check.sh`
# has to break it on purpose and watch it fail, which is only possible if the
# thing being broken is a command rather than a string inside a build script. The
# same reason `tools/symbols.sh` is a file.
#
# <header>... is the tool's table header lines, one per argument. It is a list
# rather than one string because it is the only part that differs between tables:
# a tool prints its tables under whatever headings its own output gives them, and
# the marker names are named after the step rather than after the tables, so
# nothing here can derive them. They are joined below into one anchored
# alternation, so a caller cannot leave a bracket or a `|` unbalanced. They are
# matched as extended regular expressions rather than as literals, and a header
# carrying `[`, `*` or a backslash would want escaping; none of the three
# `scale-profile` prints does.
#
# <floor> is the smallest extraction this check will accept, and it exists because
# `cmp -s` exits 0 on two empty files. See the floor below for the whole argument.
#
# Exits 0 when the two sides are byte-identical and nothing is printed; 1 when
# they are not, or when either side could not be read; 2 on a usage error.
set -eu

if [ "$#" -lt 5 ]; then
    echo "usage: sh tools/table-block.sh <tool-output> <readme> <block> <floor> <header>..." >&2
    exit 2
fi

tool=$1
readme=$2
block=$3
floor=$4
shift 4

# A floor of 0 is the one caller mistake the floor cannot defend against: every
# extraction is then "long enough", which is exactly the vacuous pass this
# script exists to stop, and it would be silent. A non-integer is refused for
# the same reason -- `[ "$lines" -lt "$floor" ]` on a non-numeric floor is an
# error, and an error that only fires on one code path is not a check.
case "$floor" in
    '' | *[!0-9]*)
        echo "table-block: floor must be a whole number, got '$floor'" >&2
        exit 2
        ;;
esac
if [ "$floor" -lt 1 ]; then
    echo "table-block: floor is 0, so every extraction would be 'long enough' and" >&2
    echo "an empty one would pass -- the vacuous gate this check exists to stop." >&2
    echo "Pass the smallest line count that is still a table; the shipped call" >&2
    echo "uses 20 against a real 29." >&2
    exit 2
fi

# The alternation, assembled here rather than in the caller so that the call
# site is a list of the tool's own header lines and not one long string somebody
# has to keep anchored and balanced.
re='^('
sep=''
for header in "$@"; do
    re="${re}${sep}${header}"
    sep='|'
done
re="${re})\$"

t=$(mktemp)
r=$(mktemp)

# The extraction is two rules and no more. A header line opens a table and a
# blank line closes it, and the blank between two tables is emitted ahead of the
# second so neither side can differ by a trailing newline. It keys on the header
# LINES rather than on line numbers, so a row moving or a column appearing reads
# as a diff rather than as a misread.
awk -v re="$re" '
  $0 ~ re {hdr=1}
  hdr {if (n++) print ""; on=1; hdr=0}
  on && /^$/ {on=0; next}
  on {print}
' "$tool" > "$t"

# A floor on the tool side, before the `cmp`, because `cmp -s` exits 0 on two
# empty files. Delete the block in the README, rename one header line in the tool,
# and the extraction finds nothing: the comparison then succeeds on nothing at
# all. That is the same defect as the `sed` pipeline the other platform had,
# where an empty derivation becomes a silently wrong `-arch` flag.
#
# What the floor MEANS, since a shared script cannot know the size of an
# arbitrary table: it is the number of lines below which an extraction is treated
# as having found nothing rather than as a table to diff. Set it well clear of
# zero and well under the real length, so that adding a row to a table does not
# require editing this check -- 20 against a real 29 for scale-profile, which is
# three headers, 24 rows and two blank separators.
#
# The README side carries no floor of its own, and that survives being
# generalised: an emptied README block is still compared against the NON-empty
# tool extraction, so `cmp` fails on the difference. What it does need is the
# marker test below, which is a different thing -- see there.
lines=$(wc -l < "$t")
lines=$((lines)) # macOS wc pads its count; the arithmetic drops the padding
if [ "$lines" -lt "$floor" ]; then
    echo "table-block: the '$block' block -- the tool printed $lines lines of table," >&2
    echo "where this check requires at least $floor, so the comparison below would" >&2
    echo "be running against an extraction that found nothing and passing on it." >&2
    echo "Either the tool printed nothing at all, or a table header line in it no" >&2
    echo "longer matches the one this awk keys on:" >&2
    echo "  $re" >&2
    echo "Run the tool and read what it prints, then re-copy the block between" >&2
    echo "<!-- $block:begin --> and <!-- $block:end -->." >&2
    rm -f "$t" "$r"; exit 1
fi

# The markers themselves, before the block is extracted. This is NOT the length
# floor above and it is not a substitute for it: a deleted marker yields an empty
# extraction, which `cmp` already fails on against a non-empty tool side, so the
# check does not silently skip -- but it fails with a diff as long as the whole
# table, and sometimes as long as the rest of the file, saying nothing about which
# of the two things happened. A deleted block is a re-copy; a deleted marker is a
# step somebody removed, and the second has to say so in those words.
#
# Found by `tools/table-block-check.sh`, whose second case deletes a marker and
# found the diff rather than the cause.
if [ ! -r "$readme" ]; then
    echo "table-block: the README holding the '$block' block is unreadable ($readme)." >&2
    echo "A gate that cannot read its subject has not passed it." >&2
    rm -f "$t" "$r"; exit 1
fi
for edge in begin end; do
    if ! grep -qF "<!-- $block:$edge -->" "$readme"; then
        echo "table-block: the README has no <!-- $block:$edge --> marker, so there is no" >&2
        echo "'$block' block to compare against and this check would be skipped rather" >&2
        echo "than graded. Put the markers back around the quoted tables, or delete" >&2
        echo "the step." >&2
        rm -f "$t" "$r"; exit 1
    fi
done

awk -v b="$block" '
  index($0, "<!-- " b ":begin -->") {on=1; next}
  index($0, "<!-- " b ":end -->") {on=0; next}
  on && /^```/ {next}
  on {print}
' "$readme" > "$r"

if cmp -s "$t" "$r"; then rm -f "$t" "$r"; exit 0; fi

# The markers are named and the diff is printed, so the fix is a copy-paste
# rather than an investigation.
echo "table-block: the '$block' block in the README is not what the tool prints:" >&2
echo "re-copy the block between <!-- $block:begin --> and <!-- $block:end -->" >&2
diff -u "$t" "$r" >&2 || true
rm -f "$t" "$r"
exit 1