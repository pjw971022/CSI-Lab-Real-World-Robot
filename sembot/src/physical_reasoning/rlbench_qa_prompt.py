task_planner = """<Instruction>
The context <C> is provided by the previously executed plan, feedback from the environment, and the Q/A history history up to this point.
You need to generate plan <P> based on a given task <T> and context <C>. <P> is the plan you need to perform next.
</Instruction>

<Examples>
<T> Could you please set up the fork for the steak for me? The task involves precisely positioning a fork next to a plate already containing a steak. </T>
<O> plate, steak, fork, knife, spoon </O>
<C> 
Execution history: [grasp the fork, back to default pose, move to 10cm to the right of the plate],
Q/A history: [Is the fork clean and ready for use?, Is there adequate space next to the plate for the fork?],
Environment feedback history: [Fork grasped successfully, Moved to correct position beside the plate]
</C>
<P> open gripper </P>

<T> place the blue block on the yellow block, making sure to navigate around a mug filled with hot coffee, which is precariously close on the same cluttered tabletop. </T>
<O> blue block, yellow block, mug </O>
<C>
Execution history: ["grasp the blue block while keeping at least 15cm away from the mug", "back to default pose"],
Q/A history: ["Has the blue block been secured without touching the mug?"],
Environment feedback history: ["Blue block grasped with sufficient clearance"]
</C>
<P> move to 5cm on top of the yellow block while keeping at least 15cm away from the mug </P>

<T> Open the drawer slowly and quietly, being mindful not to create noise that could disturb the tranquility of the library setting, where others are reading just a few feet away. </T>
<O> airpods, drawer </O>
<C>
Execution history: ["grasp the drawer handle, at 0.5x speed"],
Q/A history: ["Was the handle grasped firmly?"],
Environment feedback history: ["Drawer handle grasped at reduced speed"]
</C>
<P> move away from the drawer handle by 25cm, at 0.5x speed </P>

<T> Place the grape in the tray containing bread while ensuring not to disrupt other kitchen activities, including the ongoing use of drills and routers on the same counter for a home improvement project. </T>
<O> grape, lemon, drill, router, bread, tray </O>
<C>
Execution history: ["grasp the grape", "back to default pose"],
Q/A history: ["Has the grape been damaged during grasping?"],
Environment feedback history: ["Grape grasped and held securely"]
</C>
<P> move to the top of the tray that contains the bread </P>

<T> Carefully unplug the charger from the wall socket located behind a small, cramped home office desk, ensuring the cable is free from tangles and not obstructed by nearby furniture. </T>
<O> charger, outlet </O>
<C>
Execution history: ["grasp the charger"],
Q/A history: ["Is the charger firmly grasped?"],
Environment feedback history: ["Charger securely grasped"]
</C>
<P> back to default pose </P>

<T> Pass me a tissue and place it next to the bowl on a dining table, being careful not to disturb any other dining paraphernalia or encroach on the meal setup. </T>
<O> tissue box, tissue, bowl </O>
<C>
Execution history: ["grasp the tissue", "back to default pose", "move to 10cm to the right of the bowl"],
Q/A history: ["Is the tissue intact after being grasped?", "Is the placement beside the bowl clear of any items?"],
Environment feedback history: ["Tissue moved beside the bowl"]
</C>
<P> open gripper </P>

<T> Sweep the marbles into the tray using the broom, ensuring meticulous care to collect all small pieces from a child-friendly play area carpet, where children frequently play and could easily miss a marble with bare eyes. </T>
<O> marbles, tray, broom </O>
<C>
Execution history: ["grasp the broom", "back to default pose"],
Q/A history: ["Was the broom handle grasped securely?"],
Environment feedback history: ["Broom is now in default position ready for use"]
</C>
<P> push the marbles into the tray </P>


<T> Place the sour lemon into the top drawer, maneuvering around a busy kitchen station that includes frequently used utensils and high-traffic areas, ensuring the drawer opens and closes without snagging any hanging towels or tools. </T>
<O> orange, QR code, lemon, drawer </O>
<C>
Execution history: ["grasp the top drawer handle", "move away from the top drawer handle by 25cm", "open gripper", "back to default pose", "grasp the lemon"],
Q/A history: ["Was the drawer handle maneuvered without issues?", "Has the lemon been grasped securely?"],
Environment feedback history: ["Drawer handle released, lemon secured"]
</C>
<P> move to 10cm on top of the top drawer </P>


<T> Open the fridge door carefully, making sure to avoid any disturbance to the hot soup placed precariously close on the kitchen counter, which could pose a burn hazard. </T>
<O> fridge, hot soup </O>
<C>
Execution history: ["grasp the fridge handle and keep at least away from the hot soup", "move away from the fridge handle by 25cm and keep at least 15cm away from the hot soup"],
Q/A history: ["Has the fridge handle been grasped with the correct orientation?", "Is the path clear of any obstacles including the hot soup?"],
Environment feedback history: ["Fridge handle manipulated with clearance"]
</C>
<P> open gripper </P>


<T> Close the drawer in an office environment, ensuring all sensitive documents remain unscuffed and undisturbed during the closure process. </T>
<O> drawer, umbrella </O>
<C>
Execution history: ["push close the drawer handle by 25cm"],
Q/A history: ["Was the drawer initially open to an adequate distance?"],
Environment feedback history: ["Drawer pushed towards closed position"]
</C>
<P> continue pushing until fully closed </P>

<T> Move to the top of the cyan bowl, navigating around a kitchen counter decorated with a vibrant array of bowls and utensils, ensuring not to disrupt any other kitchenware. </T>
<O> cyan bowl, yellow bowl, box, ice cream </O>
<C>
Execution history: [],
Q/A history: [],
Environment feedback history: []
</C>
<P> move to the top of the cyan bowl </P>


<T> Turn off the lamp using the switch, ensuring the movement is gentle to maintain the calm ambiance of a dimly lit reading room. </T>
<O> lamp, switch </O>
<C>
Execution history: ["close the gripper", "move to the center of the switch", "back to default pose"],
Q/A history: ["Was the switch toggle accessible?", "Did the gripper properly interact with the switch?"],
Environment feedback history: ["Switch toggled successfully, lamp turned off"]
</C>
<P> done </P>

<T> Close the beer by turning the cap clockwise, ensuring the action is quick and secure to avoid spills in a busy bar setting where space is limited and patrons are close by. </T>
<O> beer </O>
<C>
Execution history: ["grasp the beer cap", "turn clockwise by 180 degrees", "back to default pose"],
Q/A history: ["Was the cap on securely before turning?", "Did the cap turn the full 180 degrees?"],
Environment feedback history: ["Cap turned and secured"]
</C>
<P> done </P>

<T> Take the steak out of the grill and place it flat on the plate, ensuring careful handling to maintain the steak's presentation amidst the hustle and bustle of a busy restaurant kitchen. </T>
<O> steak, grill, plate </O>
<C>
Execution history: ["grasp the steak", "back to default pose", "rotate the gripper to be 45 degrees slanted relative to the plate", "move to 10cm on top of the plate", "open gripper", "back to default pose"],
Q/A history: ["Was the steak grasped without damaging its texture?", "Was the placement on the plate smooth and centered?"],
Environment feedback history: ["Steak removed from grill and placed on plate successfully"]
</C>
<P> done </P>

</Examples>
"""

embodied_task_prompt_with_examples_last = """
Input:
<T> {input} </T>
<O> {objects} </O>
<C> Execution history:: [{execution_history}], Q/A history: [{QA}], Environment feedback history: [{feedback}]</C>
Output:
"""
def get_task_planner_prompt(instruction, objects, execution_history='', qa='', feedback=''):
    prompt = task_planner
    prompt += embodied_task_prompt_with_examples_last.format(input=instruction, objects=objects, execution_history=execution_history, QA=qa, feedback=feedback)
    return prompt