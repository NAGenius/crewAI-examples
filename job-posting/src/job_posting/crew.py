from typing import List
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

# Check our tools documentations for more information on how to use them
from crewai_tools import SerperDevTool, FileReadTool
from pydantic import BaseModel, Field
from crewai.agents.agent_builder.base_agent import BaseAgent

seper_dev_tool = SerperDevTool()
file_read_tool = FileReadTool(
    # file_path='job_description_example.md',
    file_path='E:/Project/crewAI-examples/job-posting/src/job_posting/job_description_example.md',
    description='A tool to read the job description example file.'
)

# llm = LLM(
#     model='',
#     base_url='',
#     temperature=0.7,
# )

llm1 = LLM(
    model="gemini/gemini-2.0-flash", # call model by provider/model_name
    temperature=0.7,
    # stream=True,
)

class ResearchRoleRequirements(BaseModel):
    """Research role requirements model"""
    skills: List[str] = Field(..., description="List of recommended skills for the ideal candidate aligned with the company's culture, ongoing projects, and the specific role's requirements.")
    experience: List[str] = Field(..., description="List of recommended experience for the ideal candidate aligned with the company's culture, ongoing projects, and the specific role's requirements.")
    qualities: List[str] = Field(..., description="List of recommended qualities for the ideal candidate aligned with the company's culture, ongoing projects, and the specific role's requirements.")

@CrewBase
class JobPostingCrew:
    """JobPosting crew"""
    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def research_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['research_agent'], # type: ignore[index]
            tools=[seper_dev_tool],
            llm=llm1,
            verbose=True
        )
    
    @agent
    def writer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['writer_agent'], # type: ignore[index]
            tools=[seper_dev_tool, file_read_tool],
            llm=llm1,
            verbose=True
        )
    
    @agent
    def review_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['review_agent'], # type: ignore[index]
            tools=[seper_dev_tool, file_read_tool],
            llm=llm1,
            verbose=True
        )
    
    @task
    def research_company_culture_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_company_culture_task'], # type: ignore[index]
            agent=self.research_agent()
        )

    @task
    def research_role_requirements_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_role_requirements_task'], # type: ignore[index]
            agent=self.research_agent(),
            output_json=ResearchRoleRequirements
        )

    @task
    def draft_job_posting_task(self) -> Task:
        return Task(
            config=self.tasks_config['draft_job_posting_task'], # type: ignore[index]
            agent=self.writer_agent()
        )

    @task
    def review_and_edit_job_posting_task(self) -> Task:
        return Task(
            config=self.tasks_config['review_and_edit_job_posting_task'], # type: ignore[index]
            agent=self.review_agent()
        )

    @task
    def industry_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['industry_analysis_task'], # type: ignore[index]
            agent=self.research_agent(),
            output_file='job.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the JobPostingCrew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )