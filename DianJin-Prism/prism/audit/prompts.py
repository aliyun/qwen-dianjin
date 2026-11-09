
from __future__ import annotations

import re


_VERDICT_CONTRACT_TEXT = (
    "Visible reply must be exactly one JSON object:\n"
    "{\n"
    '  "quote": "If invalid, copy one span of SOLUTION TEXT character for character — do not '
    'paraphrase or describe, it must be findable by exact search. Long enough to occur only once '
    "(one line is usually enough). When the error is that something is missing or never used, "
    "quote the line where that item DOES appear. Write null when no single span carries the fault "
    "(e.g. the complaint is about how the whole response is arranged, or the fault only exists "
    'between two places far apart); never invent a span. Empty string when valid.",\n'
    '  "reason": "<=2 sentences. If invalid, mention only the first genuine error with '
    'step/location; if valid, summarize key checks.",\n'
    '  "valid": true | false\n'
    "}"
)

_VERDICT_CONTRACT_IMAGE = (
    "Visible reply must be exactly one JSON object:\n"
    "{\n"
    '  "reason": "<=2 sentences. If invalid, cite only the first genuine visual error with '
    'image number and visible evidence; if valid, summarize key visual risks checked.",\n'
    '  "valid": true | false\n'
    "}"
)

_DESC_CONTRACT = (
    "After your thinking, your VISIBLE reply must be EXACTLY the block below, and nothing else:\n"
    "<description>\n"
    "...\n"
    "</description>"
)



TEXT_VERDICT_TEMPLATE = """You are a strict reviewer auditing ONLY the text of an interleaved text+image \
solution. Images are not shown. Your job is to find genuine content and logic errors. Apply every rule \
below literally.

TASK:
{prompt}

SOLUTION TEXT (verbatim, unprocessed model output):
{text_chain}

Highest-priority rules:
1. Explicit exemptions override defect claims. If a suspected issue falls under an exemption, \
it must not be flagged, even if it still feels wrong to you.
2. Fail fast: the first genuine content error that survives rule 1 decides valid=false. Do \
not list a second error. If nothing survives, valid=true.

Structure and notation:
The text contains the model's own planning and step-by-step reasoning alongside the answer \
itself; a trace that mixes the two is normal and expected. Section headings, step labels, and \
the wording of image descriptions vary by task type. Notation: a bare <image> marks one \
actually generated image, while <imageN> is a plain-text reference label. None of this is what \
you judge — take the layout as given and go straight to the content checks below.

Exemptions (each of these is NEVER an error):
- The opening or first step previews the goal, the final result, or the story direction.
- The final step, final image directive, final slot, or final answer contains the final result.
- An image directive quotes a correct result that the chain does derive somewhere.
- A recurring anchor object or character is absent from some scenes or steps.
- Minor differences between narrative and image directive (pose, held object, placement, camera \
angle, lighting, wording).
- Explicit flashbacks, memories, dream sequences, cutaways, or announced time jumps.
- Rounding, approximation, or restating the same correct value multiple times.
- Anything about output format, tone, wordiness, tag conventions, or marker counts. This exemption \
covers presentation only; it does NOT excuse a step or scene that carries no actual content (see \
the missing-substance check below).
- Facts from outside world knowledge that the task itself never states (do not import cooking \
science, physics trivia, or brand facts to fail a sample).

What counts as a genuine content error (check each explicitly):
- Wrong final answer: the final answer does not match what the chain derived, or answers a \
different question than the task asked.
- Time reversal: the story says evening/night/holiday-peak has arrived, and a LATER step goes \
back to afternoon/preparation/eve WITHOUT an explicit flashback marker.
- State regression: something already completed, planted, cooked, grown, or assembled appears \
later in an earlier, undone state without explanation.
- Unfinished main line: the chain announces N stages/scenes/steps or a goal action (plant the \
tree, bake the cake) and the text ends without delivering it.
- Internal contradiction inside one step or one directive: physically or semantically impossible \
combinations, e.g. "seen from behind" while "moving toward the viewer", or the same step calling \
an item both raw and cooked.
- Cooking/tutorial specific — run BOTH of these scans:
  (a) Ingredient-use scan: read the ingredient/tool list, then confirm each listed item is \
actually used in some later step. A listed item that never gets used is an error.
  (b) Premature-finish scan: for every step BEFORE the process completes, its narrative and its \
image directive must describe the CURRENT unfinished state; if a prep or intermediate step's \
directive describes the finished, plated, garnished, or fully-cooked dish, that is an error \
(the opening preview and the final step are exempt).
- Task constraint violation: an explicit requirement of the task (quantity, scene count, step \
count, required object/character, required process) is not met by the CONTENT.
- Missing substance in an announced step or scene: the step or scene is present but carries no \
usable content — a tutorial step that names an action without the instruction needed to carry it \
out, or a story scene with no narrative event. Judge substance, not word count: a terse step that \
a reader can still follow or picture is fine; flag it only when the content a reader needs is \
absent. If the response reduces MOST of its steps or scenes this way, that is this error.

First-error stop rule:
Return valid=false citing exactly the first genuine error you confirmed, with its step or \
location. Otherwise return valid=true.

{contract}
"""


TEXT_VERDICT_TEMPLATE_NOTHINK = """You are auditing ONLY the visible text of a no-think interleaved text-and-image response. \
Images are audited separately. Judge whether the written content and bare <image> marker structure \
satisfy the task.

TASK:
{prompt}

SOLUTION TEXT:
{text_chain}

Universal no-think contract:
- A bare <image> marker means an image is generated at that position. Treat draw/show/illustrate/ \
picture/image requirements as satisfied by nearby bare <image> markers for text-review purposes.
- Do not fail the text because pixels are absent, visual quality is unknown, or the image may not \
match the text; those are image-review issues.
- Do not require hidden reasoning, <think> tags, numbered image references, a separate final \
summary, or a rigid output layout unless the task explicitly asks for that exact text.
- If the task says to say/name/state/describe something after a drawing, accept the information \
when it is clearly attached to the same image unit: before the marker, after the marker, or in \
the same paragraph/section. Do not require a rigid label, separate sentence, or literal post-marker \
position.
- Future-tense or assistant-style phrasing such as "I will create" is acceptable when the required \
content is still clear.

Universal checks:
1. Marker count: enforce only explicit image-count requirements. "Exactly one image" requires one \
bare <image>; "N images/views/scenes/steps/illustrations" requires N bare <image> markers unless \
the task explicitly allows otherwise.
2. Required verbal facts: any names, labels, directions, emotions, colors, counts, object identities, \
or other facts the task explicitly asks to say in words must be stated clearly near the relevant \
marker or in a clear overall sentence.
3. Required content coverage: the text must mention the required objects, stages, relations, or \
scene changes at a usable level. Short or generic phrasing is acceptable only if it still covers \
the task's explicit requirements.
4. Logical consistency: fail only clear text-side contradictions, skipped main events, impossible \
orders, or unexplained regressions. Do not over-read harmless detail.
5. Stop-after-image constraints: if the task explicitly says "exactly one image" and "stop after" \
generating it, text after the single <image> marker must not introduce another scene, next step, \
or second visual plan.

Task-family addenda, applied only when relevant:
- Single-image description or attribute tasks: one marker plus a concise description of the \
required subject/attribute is enough; no step-by-step reasoning is needed.
- Recognition tasks: the text must provide the requested names or identities; the names may appear \
before or after the corresponding marker.
- Emotion or perception tasks: if each scene clearly names or describes the requested emotion or \
perception, do not require a separate post-image label.
- Localization, position, or before/after tasks: the text must state the requested locations or \
changes for the relevant states. It need not include a separate final answer if the states are \
already clear.
- Quantity tasks: explicit requested counts in the text must match the task. If the task asks for \
a changing count, the written sequence should not contradict that change.
- Process, lifecycle, story, weather, or lighting tasks: check only for clear ordering errors, \
missing required stages, extra stages that violate an explicit count, or state regression. Repeated \
or near-duplicate wording is not a text error by itself if the main progression remains clear and \
no required state is contradicted. Do not fail for missing a minor descriptive detail or for \
harmless continuity wording when the required moment is otherwise covered.
- Tutorial or recipe tasks: a cover/final-result preview, ingredient overview, setup image, or \
style note before the main instructions is acceptable. "Each key step" means each main \
instructional action should have a nearby image marker; it does not require every introduction, \
ingredient list, overview sentence, or style note to have its own marker. A marker may appear \
before or after its instruction if the pairing is clear. Do not fail for extra helpful overview \
images unless they replace a required key step, make the order incoherent, or contradict an \
explicit exact image count. Flag missing listed ingredients/tools only when they are clearly never \
used, counting synonyms, grouped mentions, toppings, garnish, and combined actions as use.

Return valid=false only for concrete text-side errors under these rules, and list the errors. \
Otherwise return valid=true.

{contract}
"""


IMAGE_VERDICT_TEMPLATE = """You are a strict reviewer auditing ONLY the image sequence of an interleaved \
solution — the visual state progression for the task. Step text is not shown. You see {n_images} \
images in order, step 1 through step {n_images}. Judge only what is actually visible in the \
pixels. Never assume the expected content is present: if you did not clearly see it, you may not \
claim it. Apply every rule below literally.

TASK:
{prompt}

Universal rules:
Your goal: decide whether this image chain delivers the progression the TASK calls for.
1. Pixel first. To call anything wrong OR right, you must have actually seen it. Claiming a \
diagram "correctly shows" required numbers or counts without clearly reading each one is \
fabrication and a review failure.
2. To return invalid, cite one concrete image number and the concrete visible evidence.
3. Fail fast: the first genuine visible error decides valid=false; do not list a second one. If \
no rule below is violated, valid=true.
4. First determine from the TASK whether the image count is strict or flexible. Exact-count \
constraints are strict: phrases like "exactly one", "one image", "one per", "draw each", or \
enumerated Scene 1..N / Step 1..N require that exact number of visual units. Loose stage requests \
such as "at three stages", "through the process", or "step by step" may include extra \
intermediate images only if they form a clear, non-redundant progression and do not replace or \
obscure a required stage. For flexible stage requests, repeated near-identical states or decorative \
extra endings that do not add a real stage are invalid.
5. For every image in order, check the required objects, count, colors/attributes, spatial \
relations, action/state, weather/lighting condition, and whether it advances from the previous \
image. A chain is valid only if all required stages are present, ordered, and visually distinct.
6. The FIRST image may be a goal/cover/story preview and the LAST image may show the final \
state only when that does not conflict with an explicit required stage. For tutorials or recipes, \
a first cover image, final-result preview, ingredient overview, setup image, or prepared-looking \
component overview is acceptable; do not treat it as state regression if later images return to \
ingredients or preparation, as long as the main instructional sequence then progresses coherently.
7. These are NEVER errors by themselves: slight face/hair/style drift, texture noise, lighting/ \
shadow/warm-light color shifts, two similar but DISTINCT props, background bystanders changing, \
an anchor object missing from some scenes, clothing changes between scenes, symbolic or \
decorative flourishes, or a photographic rather than hand-drawn style. Words like draw/show/ \
illustrate describe the generation task, not a required art style, unless the task explicitly asks \
for sketch, cartoon, line art, or hand-drawn style. These exemptions do NOT excuse wrong counts, \
missing required objects, unchanged before/after states, state regression, skipped stages, or \
extra unrelated stages.
8. If the task says to say/name/state/describe something after a drawing, that verbal requirement \
is audited by text review, not image review. Do NOT require text labels or captions inside the \
image unless the task explicitly asks for readable written text in the pixels, sign reading, OCR, \
or exact words. For cards, labels, screens, and signs in non-OCR tasks, visible interaction or \
context is enough; illegible small text is not an error by itself.

What counts as a genuine visible error (check each; these are the dominant real failures):
- No task progress: an image repeats the same required moment, changes only camera angle/style/ \
background, or adds unrelated decoration instead of advancing the task's required state.
- Wrong image/stage count: a strict exact-count task has the wrong number of visual units, or a \
flexible stage task adds images that repeat a previous state, add an unrelated or merely decorative \
ending, reverse the state, make the required progression ambiguous, or replace/obscure a required \
stage.
- Wrong quantity: a required visible count is wrong, non-monotonic when the task asks for \
increase/decrease, or only implied by clutter rather than clearly visible objects.
- Failed before/after or swap: the required change does not actually happen, objects remain in \
the same state/place, the direction is reversed, or the images merely show another viewpoint.
- Weather/lighting/state mismatch: the requested condition is absent, appears in the wrong stage, \
loops backward, or extra day/night/weather states are inserted beyond the requested sequence.
- Corrupted key subject: a face, body, or main object that is melted, duplicated, or \
unrecognizable, or a main structure that has visibly collapsed into incoherence.
- Premature final state: in a task that shows a progression, an intermediate image already \
shows the finished result while the process is still mid-way. Early images must show the early, \
unfinished state.
- State regression or identity swap: something already advanced appears reverted to an earlier \
state, or a key object or tool becomes a different kind of object between images.
- A single image that is internally impossible or self-contradictory in an obvious visible way.
- The image visibly fails to convey the one thing its step exists to show.

First-error stop rule:
Return valid=false citing exactly the first genuine visible error (image number + what you \
see). Otherwise return valid=true.

{contract}
"""


TEXT_DESC_TEMPLATE = """Rewrite the solution into exactly {n_images} image-generation prompts, one per image, in order. Think in one or two short lines, then write the slots. Do not re-list or check these rules in your thinking; just write.

TASK:
{prompt}

ORIGINAL TEXT (contains {n_images} image markers, in order):
{text_chain}

REVIEWER NOTE (a clue, may be empty): {findings}

Rules:
1. Output exactly {n_images} blocks: <slot 1> ... </slot 1> through <slot {n_images}> ... </slot {n_images}>, in the SAME order as the original image markers. Do NOT reorder the slots, even if the sequence looks illogical (e.g. a finished-dish or goal preview appears before the prep/steps) — that original order is intentional; keep each slot at its original position. Exactly ONE slot per image marker: do not add, split, merge, or drop — the slot count is exactly {n_images} (even when {n_images} is 1, output exactly 1 slot).
2. Each block contains ONE thing only: a concrete text-to-image prompt for that step's image — name the key subjects with their counts and attributes, and the current visible state. Convey the step visually; do not ask for written words, labels, or numbers rendered inside the image. One to three sentences. No labels, no reasoning, no step description, no extra fields.
3. Make the prompt depict the CORRECTED state: fix clear errors from the original (wrong count or number, wrong object, a missing required element, or a step drawn in a later / finished state). Otherwise stay faithful to what that step shows.
4. A middle step's image shows its CURRENT state, never a later cooked / finished / assembled / future state. Only the first or last slot may show a preview or finished result, and only if the original did so there.
5. Do not invent new objects, characters, or outside-world details. Do not output <image>, <imageN>, or any image marker; code inserts them.
6. Keep recurring characters and objects consistent across slots, but do NOT re-describe their full appearance every slot. Identify them with a short tag (e.g. "the same boy in the green jacket") and spend each slot mainly on what is NEW or changed this step — the new action, setting, pose, or state — so consecutive images progress rather than repeat.

FEW-SHOT EXAMPLE:

Example input:
TASK:
Show, step by step, how to repot a houseplant, using interleaved text and images. Include 2 illustrations, one per step.

ORIGINAL TEXT:
First, gently lift the plant out of its old pot and loosen the tangled roots.
Picture for this step: the plant sitting neatly in a clean new pot with fresh soil, on a bright windowsill.
<image1>
<image>
Then settle the plant into the new pot and firm fresh soil around the roots.
Picture for this step: the same plant upright in the new pot with fresh soil patted down, on the same windowsill.
<image2>
<image>
Final answer: the plant is repotted and ready to water.

REVIEWER FINDINGS:
Step 1 is drawn in the finished state: it already shows the plant potted in the new pot, though this step is only lifting it out and loosening the roots.

Example output:
<description>
<slot 1>
A pair of hands lifting a leafy green houseplant out of its old terracotta pot on a bright windowsill, the exposed root ball still tangled and clinging to dark soil, the empty new pot waiting to one side.
</slot 1>
<slot 2>
The same houseplant now settled upright in the new pot on the same windowsill, fresh soil patted firmly around its base, loose crumbs of soil on the sill.
</slot 2>
</description>

Now output only the <description> block with exactly {n_images} slots:
"""


TEXT_DESC_TEMPLATE_NOTHINK = """Rewrite the image prompts for a no-think interleaved response. The original visible text is mostly usable, but the images were judged faulty and the text around a marker may contain a faulty image directive. Produce corrected image prompts, one slot per original <image> marker, preserving the intended meaning and order while fixing the visual-side error.

TASK:
{prompt}

ORIGINAL VISIBLE TEXT (contains {n_images} image markers, in order):
{text_chain}

REVIEWER NOTE (a clue, may be empty): {findings}

Rules:
1. Output exactly {n_images} blocks: <slot 1> ... </slot 1> through <slot {n_images}> ... </slot {n_images}>, in the SAME order as the original image markers. Exactly ONE slot per image marker; do not add, merge, split, reorder, or drop slots.
2. Each slot is the corrected image prompt for the corresponding original marker. Read the surrounding ORIGINAL VISIBLE TEXT, keep its intended subject, action, count, attribute, location, and role in the sequence, and correct only the visual-side error.
3. Write a concrete visual prompt for the CURRENT image: required subjects, counts, attributes, spatial relations, action/state, and any required progression cue. One to three sentences is usually enough.
4. If the task requires readable text, signs, labels, letters, numbers, OCR, diagrams, or named items visible in the image, explicitly request those visible words/numbers. Otherwise do not invent unnecessary writing.
5. For tutorials, recipes, processes, stories, and lifecycle tasks, each slot should match the current stage around that marker. A cover/final-result preview or ingredient/setup image is allowed only if the original text places that image there.
6. Do not invent new objects, characters, settings, or outside-world details. Do not output <image>, <imageN>, or any image marker; code inserts them.
7. Keep recurring characters and objects consistent across slots, but focus each slot on what is new or changed at that marker.

Example input:
TASK:
A bicycle shop changed its display bikes. Bike one was RED and bike two was BLUE. The shop changed them: bike one became YELLOW and bike two became GREEN. Draw the shop before, then after. For each drawing, say in words the color of each bike.

ORIGINAL VISIBLE TEXT:
The shop's display before the change: Bike one is RED, and bike two is BLUE.
<image>
The shop's display after the change: Bike one is YELLOW, and bike two is GREEN.
<image>

REVIEWER NOTE:
In the final image, both bicycles are yellow, whereas the task requires the second bike to become green.

Example output:
<description>
<slot 1>
A bicycle shop display before the change, showing exactly two display bikes side by side: bike one is red and bike two is blue, with both colors clearly visible and distinct.
</slot 1>
<slot 2>
The same bicycle shop display after the change, showing exactly two display bikes side by side: bike one has become yellow and bike two has become green, with the green second bike clearly visible and not yellow.
</slot 2>
</description>

Now output only the <description> block with exactly {n_images} slots:
"""

IMAGE_DESC_TEMPLATE = """You are writing the text of an interleaved text-and-image response. The \
{n_images} image(s) you are shown are the CORRECT visual sequence for the task below — treat them as \
ground truth. The text that originally accompanied them was faulty and has been discarded; write a \
fresh, complete response whose facts follow the images.

TASK:
{prompt}

A reviewer noted this error in the discarded text (hint only, may be imperfect): {findings}

Think in one or two short lines, then write the sections. Do not re-list or check these rules in your \
thinking; just write.

Write the response as {n_images} sections plus a summary, using exactly this skeleton:
## Image 1
<planning for image 1, in your thinking voice>
<the heading you use before an image>
<that image's own prompt>
## Image 2
<...continue through image {n_images}...>
## Summary
<the final user-facing answer>

Rules:
- Write in the task's own genre and voice: each "## Image K" section opens with your working notes in \
the response's thinking voice — brief scene/step planning that leads to that image (what happens now, \
why it follows from the previous scene) — and that planning is not a caption, so do not write "The \
image shows..." there. The image's own prompt, under the heading, is where the plain visual \
description belongs.
- Ground every fact in the images: who is present, object counts and attributes, states, locations and \
the order of events must match what the images actually show. Do not invent people, objects or events \
that appear in neither the images nor the task.
- Keep recurring characters, objects and settings consistent across sections, and connect the sections \
(what changed since the previous scene, and why it leads to the next).
- Avoid repeating the error in the reviewer hint, but verify the hint against the images; the images win.
- Use exactly {n_images} "## Image K" heading(s), numbered 1..{n_images} in order; do not add, drop, \
merge, or reorder them. Never output the bare marker "<image"+">" anywhere; code inserts those.
- Under "## Summary", write the COMPLETE final user-facing answer (the full story / the full \
instructions / the full solution with its result), segmented ONE paragraph per scene or step in order, \
and END each paragraph with that scene's numbered reference tag: the first paragraph ends with \
<image1>, the second with <image2>, ... through <image{n_images}>. This matches how a finished \
response cites its illustrations.

EXAMPLE 1 — a story task. Heading before each image: "# Image Generation Prompts". Summary: plain \
prose paragraphs, no headings.

Example task: Tell a short 2-scene picture-book story (theme: losing and finding a kite).

Example output:
## Image 1
The story opens on a breezy hill: Lily is flying her red diamond kite with a yellow tail, joy building \
as it climbs — and this scene should end on the turning point, the moment a gust rips the spool from \
her hands. Key elements to keep consistent: the red kite, its yellow tail, the grassy hill, the old \
oak in the distance.
# Image Generation Prompts
A young girl in a yellow raincoat stands on a green grassy hill, both hands gripping the string of a \
red diamond kite with a long yellow tail high above her, her face tilted up and laughing, an old oak \
tree small in the distance, bright breezy afternoon light.
## Image 2
The second scene resolves the loss. Lily follows the field to the oak tree from scene 1, spots the red \
kite caught in the lower branches, and recovers it on tiptoe. Her relief and the small lesson about \
holding on tighter close the story.
# Image Generation Prompts
The same girl in the yellow raincoat stands on tiptoe beneath the old oak, reaching up to ease the red \
diamond kite with the yellow tail out of its lower branches, relief on her face, the grassy hill \
behind her in the same afternoon light.
## Summary
On a breezy spring afternoon, Lily raced up the grassy hill with her red diamond kite, its yellow tail \
snapping happily behind her. She let the string out little by little, laughing as the kite climbed \
higher — until a sudden gust tore the spool from her hands, and the kite sailed away toward the old \
oak tree.
<image1>
Lily searched the field all the way to the fence line, her heart sinking with every step. Then a flash \
of red caught her eye: the kite had settled gently in the oak's lower branches, its yellow tail \
swaying within reach. Standing on tiptoe, she eased it free, hugged it to her chest, and promised to \
wind the string twice around her wrist next time.
<image2>

EXAMPLE 2 — a tutorial task. Heading before each image: "# Explicit Prompt:" (with the colon). \
Summary: a "### Step N: ..." heading on each step after the first.

Example task: Teach me how to make garlic butter toast with an interleaved illustrated tutorial.

Example output:
## Image 1
The tutorial should open with the preparation stage: a halved baguette, softened butter in a small \
bowl, minced garlic and chopped parsley laid out on a wooden board. This first image fixes the setting \
— keep the same board and bowl visible later so the sequence reads as one continuous counter.
# Explicit Prompt:
A rustic wooden board on a kitchen counter holding a baguette sliced lengthwise into two halves, a \
small white bowl of softened butter, a little pile of minced garlic and a scattering of chopped \
parsley, soft daylight from the left, nothing cooked yet.
## Image 2
The second step delivers the result: the buttered slices toasted until the edges turn golden and the \
topping bubbles. Ending on the finished toast, still on the same wooden board, closes the loop from \
ingredients to ready-to-serve.
# Explicit Prompt:
The same wooden board, now holding the two baguette halves toasted golden at the edges, the garlic \
butter melted and bubbling across their cut faces with flecks of parsley, steam rising, the same white \
bowl empty beside them in the same soft daylight.
## Summary
Start by laying out your ingredients on a wooden board: a halved baguette, softened butter in a small \
bowl, minced garlic and a sprinkle of chopped parsley. Mash the garlic and parsley into the butter \
until evenly combined.
<image1>
### Step 2: Spread and Toast
Spread the garlic butter generously over the cut faces of the baguette, then toast the slices until \
the edges turn golden and the butter bubbles. Serve warm, straight from the board.
<image2>

Now write the response for the task above:
"""


IMAGE_DESC_TEMPLATE_NOTHINK = """You are writing the visible text of a no-think interleaved text-and-image response. The \
{n_images} image(s) you are shown are the CORRECT visual sequence for the task below — treat them as \
ground truth. The text that originally accompanied them was faulty and has been discarded; write a \
fresh response whose facts follow the images.

TASK:
{prompt}

A reviewer noted this error in the discarded text (hint only, may be imperfect): {findings}

Write exactly {n_images} markdown section(s), one per image, using this structure:
## Image 1
<the visible response text that should appear before image 1>
## Image 2
<the visible response text that should appear before image 2>
...

Rules:
- Match the no-think rollout style: visible user-facing response text only, no hidden planning, no <think> tags.
- Each image section should contain the natural text that appears immediately before that image in the assistant response. It should read like the answer itself, not like an internal image-generation prompt.
- Do NOT write internal planning phrases such as "I need to", "the objective is", "this image will depict", "the prompt should", or "camera angle". Do NOT write headings such as "# Explicit Prompt" or "Image Generation Prompts" inside a section.
- Ground every fact in the images and the task. Do not invent people, objects, counts, states, or events.
- Preserve required verbal facts from the task. If the task asks to say/name/state/describe something after a drawing, include that information in the same image section.
- For tutorials, write concise instructional steps tied to the visible stage; ingredient/setup or final-result preview text is allowed when it matches the image.
- For single-image description tasks, one concise description before the image is enough; do not add a final paragraph unless the task asks for one.
- Use exactly {n_images} "## Image K" section(s), numbered 1..{n_images}. Do not add, drop, merge, or reorder them.
- Do NOT write numbered image references such as <image1>, <image2>, <img1>, or <IMG_1> anywhere.
- Do NOT output the bare marker "<image"+">" anywhere; code inserts image-generation markers after each section.

EXAMPLE — real no-think single-image response:

Example task: A harbor had three buoys. One was red and round, one was green and cone-shaped, and one was white with a light on top. Show the buoys and describe each. You must generate exactly one image — not two, not three, exactly one. Stop after generating it.

Example output:
## Image 1
The harbor scene features three distinct buoys: a red spherical buoy, a green conical buoy, and a white buoy topped with a light. These buoys are positioned in the water to provide clear visual markers for navigation.

EXAMPLE — real no-think tutorial response style:

Example task: Teach me how to cook Potato Soup with an interleaved illustrated tutorial. Each key step must include a short instruction and one image of the CURRENT step's state.

Example output:
## Image 1
First, let's look at the goal: a rich, velvety soup with tender potatoes and fresh herbs, served in a warm ceramic bowl.
## Image 2
Before you start, gather your ingredients: two large potatoes, a carton of whole milk, butter, and fresh parsley. Having everything ready on your counter makes the process much smoother.
## Image 3
Step 1: Prep the potatoes. Wash them and peel them completely. Cut them into uniform 1.5cm cubes so they cook evenly.
## Image 4
Step 2: Sauté the potatoes. Heat a little butter in a pan over medium heat, then add the potato cubes and stir until the edges turn lightly golden.
## Image 5
Step 3: Simmer the soup. Pour in the milk and let the potatoes cook gently until they soften and the liquid becomes creamy.
## Image 6
Step 4: Mash and season. Lightly crush some of the tender potatoes into the soup, then season with salt and stir until the texture is smooth but still hearty.
## Image 7
Step 5: Finish and serve. Ladle the potato soup into a warm bowl and garnish it with chopped fresh parsley before serving.

Now write the no-think response for the task above:
"""



SPAN_REPAIR_TEMPLATE = """The images are the correct visual record for this task — treat them as \
ground truth. Below is one passage of a draft response written for the same task, and a reviewer's \
finding about it.

TASK:
{prompt}

REVIEWER FINDING:
{reason}

PASSAGE (the span to fix is marked between >>> and <<<):
{context}

Rewrite ONLY the marked span so the finding is resolved and the text agrees with the images. Keep \
the same voice, tense and level of detail as the surrounding passage, and keep it about the same \
length — everything outside the marked span stays as it is, so your replacement has to read \
continuously with it. Do not add commentary and do not output the surrounding passage.

Reply with exactly one block:
<replacement>
...the new text for the marked span...
</replacement>
"""


FINAL_ANSWER_TEMPLATE = """You are writing the final, user-facing answer that closes an interleaved \
text-and-image response. The working notes below are the response's own reasoning; the \
{n_images} illustration(s) it produced are correct. Your job is only the closing answer that the \
reader actually reads.

TASK:
{prompt}

WORKING NOTES (the reasoning this response wrote for itself):
{think_text}

A reviewer had found this problem in the previous answer, so do not reproduce it: {findings}

Read the working notes critically rather than trusting them. They are the reasoning that produced \
the faulty answer, so the same mistake may still be sitting in them: carry forward what holds up \
against the illustrations, and correct whatever the reviewer's finding exposes instead of repeating \
it.

SHAPE OF THE ANSWER YOU ARE REPLACING — match it:
{shape}

Requirements:
- Write the complete answer the task asks for, covering the whole progression in order. It stands \
on its own: the reader never sees the working notes above.
- Segment it into exactly {n_images} parts, one per illustration, in order. End each part with that \
illustration's reference tag on a line of its own: the first part ends with <image1>, the second \
with <image2>, and so on through <image{n_images}>. Every tag from 1 to {n_images} appears exactly \
once and nothing else goes on those lines.
{heading_rule}
- Do not narrate the pictures. The reader is looking at them, so write what happens or what to do, \
never "In the image you can see...". The working notes are full of such picture descriptions \
because they were written for the illustrator; take the facts from them and drop that framing.
- Do not open with a meta-plan. The working notes may start by describing the tutorial's visual \
style, photography look, or how many stages it will have; none of that belongs in the answer. \
Begin directly with the answer's real first part.
- Ground every fact in the working notes and the illustrations. Do not introduce people, objects or \
events that appear in neither.
- Write only the answer. No preamble about what you are doing, no planning notes, no image \
directives, no bare marker "<image"+">".

Below is one finished answer in the expected shape. Follow its structure and density, not its \
subject matter.

{example}

Now write the answer for the task above, with exactly {n_images} parts:
"""



def text_audit_prompt(prompt: str, text_chain: str, *, think_mode: bool = True) -> str:
    template = TEXT_VERDICT_TEMPLATE if think_mode else TEXT_VERDICT_TEMPLATE_NOTHINK
    return template.format(
        prompt=prompt.strip(), text_chain=text_chain.strip(),
        contract=_VERDICT_CONTRACT_TEXT,
    )


def image_audit_prompt(prompt: str, n_images: int) -> str:
    return IMAGE_VERDICT_TEMPLATE.format(
        prompt=prompt.strip(), n_images=n_images, contract=_VERDICT_CONTRACT_IMAGE,
    )


def text_desc_prompt(prompt: str, text_chain: str, findings: str, *, think: bool = True) -> str:
    chain = text_chain.strip()
    n = chain.count("<image>")
    template = TEXT_DESC_TEMPLATE if think else TEXT_DESC_TEMPLATE_NOTHINK
    return template.format(
        prompt=prompt.strip(), text_chain=chain, n_images=n,
        findings=(findings.strip() or "No issues found; clean and sharpen only."),
    )


def image_desc_prompt(prompt: str, n_images: int, findings: str, *, think: bool = True) -> str:
    template = IMAGE_DESC_TEMPLATE if think else IMAGE_DESC_TEMPLATE_NOTHINK
    return template.format(
        prompt=prompt.strip(), n_images=n_images,
        findings=(findings.strip() or "No single error was pinpointed; just make the text faithfully match the images."),
    )


def span_repair_prompt(prompt: str, reason: str, passage: str, span: tuple) -> str:
    start, end = span
    marked = passage[:start] + ">>>" + passage[start:end] + "<<<" + passage[end:]
    return SPAN_REPAIR_TEMPLATE.format(
        prompt=prompt.strip(), reason=(reason.strip() or "The marked span is inconsistent."),
        context=marked.strip(),
    )


_HEADING_RE = re.compile(r"^#{2,4}\s+\S.*$", re.MULTILINE)


def _answer_shape(orig_answer: str) -> tuple:
    body = orig_answer or ""
    heads = _HEADING_RE.findall(body)
    paras = [p for p in re.split(r"\n\s*\n", body) if p.strip() and not re.fullmatch(r"\s*<image\d+>\s*", p)]
    words = len(body.split())
    if heads:
        return (
            'It opened with a short introduction and then put a section heading on each step, '
            'formatted exactly like this one: "%s". It used %d such headings across %d paragraphs, '
            "about %d words in total. Use the same heading convention and the same density."
            % (heads[0].strip(), len(heads), len(paras), words)
        ), True, heads[0].strip()
    return (
        "It used no section headings at all — plain prose in %d paragraph(s), about %d words in "
        "total. Do not add headings." % (len(paras), words)
    ), False, ""


def final_answer_prompt(prompt: str, think_text: str, n_images: int, findings: str,
                        orig_answer: str) -> str:
    from .fewshot import STORY_ANSWER, STORY_TASK, TUTORIAL_ANSWER, TUTORIAL_TASK

    shape, headed, first_head = _answer_shape(orig_answer)
    if headed:
        heading_rule = (
            '- Put a section heading on every part, in exactly this style: "%s". This is required — '
            "do not merge the parts into one heading-less block." % first_head
        )
    else:
        heading_rule = "- Use no section headings; write plain prose paragraphs."
    task, answer = ((TUTORIAL_TASK, TUTORIAL_ANSWER) if headed else (STORY_TASK, STORY_ANSWER))
    return FINAL_ANSWER_TEMPLATE.format(
        prompt=prompt.strip(), think_text=think_text.strip(), n_images=n_images, shape=shape,
        heading_rule=heading_rule,
        findings=(findings.strip() or "No specific problem was pinpointed; just make the answer faithful."),
        example="Task: %s\n\nAnswer:\n%s" % (task.strip(), answer.strip()),
    )
