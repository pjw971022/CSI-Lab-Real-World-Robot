class PromptPourFromCupToCup:
    def __init__(self,) -> None:
        pass

    def get_u2c(self):
        context = """
            **Motion 1:**
            - Approach red cup
            - Initial Tilt: 0 degrees
            - Max Tilt: 45 degrees
            - Required Force: Medium
            - Distance Moved: 10 cm
            - Height: 5 cm
            - Speed: Slow
            - Repetitive Actions: No
            - Cautions: Avoid spilling liquid

            **Motion 2:**
            - Approach maroon cup
            - Initial Tilt: 45 degrees
            - Max Tilt: 0 degrees
            - Required Force: Medium
            - Distance Moved: 10 cm
            - Height: 5 cm
            - Speed: Slow
            - Repetitive Actions: No
            - Cautions: Ensure liquid flows into maroon cup
        """
        return context
    
    def get_o2c(self):
        context = """
            Action Name: Pick up red cup
            Initial Tilt(degree): 0
            Max Tilt(degree): 0
            Required Force: 10N
            Distance Moved: 10cm
            Height: 5cm
            Speed: 10cm/s
            Repetitive Actions: No
            Cautions: Avoid spilling liquid

            Action Name: Pour liquid into maroon cup
            Initial Tilt(degree): 45
            Max Tilt(degree): 90
            Required Force: 5N
            Distance Moved: 15cm
            Height: 10cm
            Speed: 5cm/s
            Repetitive Actions: No
            Cautions: Ensure liquid is poured into the maroon cup
        """
        return context
    
    def get_d2c(self):
        context = """
            Action Name: Approach object
            Initial Tilt(degree): 0
            Max Tilt(degree): 0
            Required Force: 10
            Distance Moved: 10
            Height: 10
            Speed: 10
            Repetitive Actions: No
            Cautions: None

            Action Name: Pour liquid
            Initial Tilt(degree): 45
            Max Tilt(degree): 90
            Required Force: 5
            Distance Moved: 10
            Height: 10
            Speed: 10
            Repetitive Actions: No
            Cautions: None
        """
        return context