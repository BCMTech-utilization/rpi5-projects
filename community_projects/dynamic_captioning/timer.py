import time

class Timer :
    def __init__(self) :
        self.start= time.time()
    
    def reset(self) :
        self.start= time.time()

    def record(self, message="", float_point= 4, reset=False) :
        """
        Args
            message: str
            float_point: int
        Return
            None

        Descriptions
            This method is just printing elapsed time
        """
        cur= time.time()- self.start
        print(f"{message}, {round(cur, float_point)}")
        
        if reset :
            self.reset()
