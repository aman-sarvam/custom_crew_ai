import os
from crewai import Agent, Task, Crew
from tools.search_tools import SearchTools
from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()

serper_api_key = os.getenv("SERPER_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

client = OpenAI(api_key=openai_api_key)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LANGGRAPH_EXECUTE_AGENT"


def create_agent(role, goal, verbose, memory, backstory, tools):
    agent_tools = []
    if "Google Search Tool" in tools:
        agent_tools.append(SearchTools.search_internet)
    
    return Agent(
        role=role,
        goal=goal,
        verbose=verbose,
        memory=memory,
        backstory=backstory,
        tools=agent_tools
    )

def create_task(description, expected_output, agent):
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )

def setup_crew_from_json(json_data):
    data = json.loads(json_data)
    agents = []
    tasks = []
    
    for agent_info, task_info in zip(data["agents"], data["tasks"]):
        agent = create_agent(
            role=agent_info["role"],
            goal=agent_info["goal"],
            verbose=agent_info["verbose"],
            memory=agent_info["memory"],
            backstory=agent_info["backstory"],
            tools=agent_info["tools"]
        )
        task = create_task(
            description=task_info["description"],
            expected_output=task_info["expected_output"],
            agent=agent
        )
        agents.append(agent)
        tasks.append(task)

    print("Agents:", agents)
    print("Tasks:", tasks)
    
    crew = Crew(
        agents=agents,
        tasks=tasks
    )
    
    results = crew.kickoff()
    print("Crew processing complete. Results:")
    print(results)


def get_chatgpt_response(user_goal):
    
    prompt = (
        f"You are tasked with designing a sophisticated AI crew for a project focused on '{user_goal}'. "
        "Each AI agent in the crew must have a specific role, goal, and set of capabilities, "
        "and each must be assigned a specific task. Each agent tailored to perform very specific small tasks related to the project's goals. "
        "Determine the optimal number of agents based on the project's complexity."
        "\n\nCreate a detailed plan for this AI crew in JSON format. Include the following for each agent:"
        "\n- Role: The specific role or expertise area of the agent."
        "\n- Goal: The precise objectives this agent aims to achieve directly related to the project."
        "\n- Verbose: Whether the agent should have verbose output (true/false)."
        "\n- Memory: Whether the agent should have memory capabilities (true/false)."
        "\n- Backstory: Think of the crew as a team of humans working together. Keep the backstory real."
        "\n- Tools: A list of tools, if any, needed by the agent (Only 1 tool available to choose from: Google Search Tool)."
        "\n\nFor each agent, also define a task with:"
        "\n- Description: A detailed description of the task."
        "\n- Expected Output: The expected result or output of the task."
        "\n\nEnsure the number of agents equals the number of tasks, with each agent linked to one specific task."
        "\n\nFormat the entire setup as a JSON object with keys 'agents', 'tasks', and 'number_of_agents', where each agent and their corresponding task are linked by indices."
       
    )
    print("Crew generation prompt:", prompt)

    try:
    
        response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )

       
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Sorry, I couldn't fetch a response. Please check your network or API settings."

def edit_crew_gpt(current_crew, crew_edit_message):
    
    prompt = (
        f""" 
        Following is the original AI crew of agents and tasks generated for a particular goal:
        
        {current_crew} 
        
        These are the edits given by the user:
        
        {crew_edit_message}
        
        Make these edits in the original AI crew and return the updated crew in JSON format. Strictly only return the crew. 
        
        """
    )
    print("Crew edit prompt:", prompt)

    try:
    
        response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
            
        ],
        
        model="gpt-3.5-turbo",
    )

       
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Sorry, I couldn't fetch a response. Please check your network or API settings."
    
if __name__ == "__main__":
    user_goal = input("Please describe the goal for your crew")
    gpt_crew = get_chatgpt_response(user_goal)        

    while True:
        print(gpt_crew)
        user_setup_crew = input("Do you want to 'kickoff' the crew or 'edit' the crew? (kickoff/edit): ")
        if user_setup_crew == 'kickoff':
            setup_crew_from_json(gpt_crew)
            break
        elif user_setup_crew == 'edit':
            crew_edit_message = input("Describe your edits")
            gpt_crew = edit_crew_gpt(gpt_crew, crew_edit_message)
        else:
            print("Invalid Input")


"""
Test Case: 
I am a car manufacturer in South Korea, and I want to start a business in India. 
My business requires me to put a Sim card in my car and I want to know how the regulations within India affect my business generate seven questions that I must answer to understand this better

Test Questions: 

1. What are the current regulations regarding the import and use of SIM cards in vehicles in India? 
- This question addresses the need to understand both importation laws and specific regulations surrounding the integration of SIM cards into vehicles.
2. Do I need to partner with a local telecom provider to operate the SIM cards in my cars, and what are the implications of such partnerships? 
- Partnerships with local telecom providers might be necessary, and this could have implications for operational logistics and costs.
3. Are there any specific data privacy regulations in India that pertain to data collected through SIM cards in vehicles? 
- Data privacy is crucial, especially in sectors involving technology and personal information, so understanding this will help ensure compliance.
4. What type of approval or certification is required from Indian authorities to use connected cars equipped with SIM cards on public roads? 
- Knowing the necessary certifications will help in planning the approval process for your vehicles.
5. How do consumer protection laws in India impact the services provided through the SIM cards in my cars? 
- Understanding consumer protection laws will help you align your product features and service offerings with legal requirements to protect consumers.
6. What are the implications of cybersecurity laws on the operation of SIM-enabled cars in India? 
- With cars becoming increasingly connected, cybersecurity laws will play a crucial role in shaping the security measures you need to implement.
7. Are there any upcoming regulatory changes in India that could affect the use of SIM cards in vehicles? 
- Staying ahead of potential regulatory changes can help you adapt your business model and technology in time to remain compliant and competitive.

"""
"""
Test Case: 
Imagine you're planning to launch a mobile app in India that collects and uses personal data for customizing user experiences and providing targeted advertisements. 
Here are seven essential questions you might ask a legal advisory firm to ensure your app complies with Indian regulations:

Test Questions: 
1.What are the key provisions of the Indian Personal Data Protection Bill that my mobile app needs to comply with? 
- Understanding the specific requirements regarding data collection, processing, and storage under this law is crucial.
2. How does the Information Technology Act impact the operations of my mobile app, especially regarding user data privacy and security? 
- This act includes important regulations on data protection and security that affect digital businesses.
3. Are there specific consent requirements I need to adhere to when collecting personal data from users through the app? 
- Knowing how and when to obtain user consent for data collection and use is vital for legal compliance.
4. What are the penalties for non-compliance with data protection laws in India, and how can I mitigate these risks? 
- Awareness of the consequences of legal infractions can help you prioritize compliance measures.
5. Do I need to appoint a data protection officer or any other specific roles to oversee compliance within my organization? 
- Some regulations may require appointing personnel to ensure ongoing compliance with data protection laws.
6. Are there any sector-specific regulations I need to be aware of that could affect the functionality or data handling practices of my app? 
- Sector-specific rules could influence aspects of your app’s operations, depending on its features and the data it processes.
7. What are the requirements for transferring personal data outside of India, should my app need to process data internationally? 
- If your app operates globally or uses international servers, you’ll need to comply with regulations regarding cross-border data transfer.
"""





