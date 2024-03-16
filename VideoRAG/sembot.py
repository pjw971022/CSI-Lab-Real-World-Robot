import os
pid = os.getpid()
os.system(f'taskset -pc 21,22,23,24 {pid}')
from vlm_utils import Retriever, H_Planner, L_Planner, Voxposer

# Retrieval 



class Sembot:
    def __init__(self):
        self.h_planner = None
        self.l_planner = None
        self.policy = Voxposer()
        self.retriever = Retriever()
        self.fgs_loop = 3
        self._setup()

    def _setup(self, env, general_config):
        lmps_config = general_config['lmp_config']['lmps']
        h_planner_config = lmps_config['h_planner']
        l_planner_config = lmps_config['l_planner']
        self.h_planner = H_Planner(h_planner_config)
        self.l_planner = L_Planner(l_planner_config)
        
    def run(self, obs, instruction):
        skill_seq = self.h_planner(obs, instruction)
        for skill in skill_seq:
            fg_skill, query = self.l_planner(obs, skill) 
            # query = f'현재 환경 정보: {context}에서 {fg_skill}을 수행하고 있다.' # Ego-MCQ
            for _ in range(self.fgs_loop):
                retrieved_video = self.retriever.retrieve_video(query)
                fg_skill, query = self.l_planner(obs, fg_skill, retrieved_video)

            self.policy(fg_skill)
