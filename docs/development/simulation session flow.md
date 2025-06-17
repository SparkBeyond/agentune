# Conversation Simulation Flow Design

Transform original conversations into realistic simulated ones by 
1. Using intent from real conversations as a seed for scenario generation
2. Generating domain specific conversation by letting the simulated participants (agents and customers) be aware of similar real conversation

## **Overview Diagram**

![Conversation Simulation Flow Diagram](./Conversation%20Simulation%20Flow%20Diagram.svg)

## Flow Steps

### Scenario Generation
- **Input**: List of original conversations
- **Process**: 
  - Extract intents from each conversation
  - Get the first message of the conversation
  - Generate scenarios based on extracted intents and the first message
- **Output**: List of scenarios ready for simulation
### Conversation Execution
- **Input**: List of scenarios
- **Process**: 
  - For each scenario, create participants (agents and customers)
  - Run the conversation simulation using the `FullSimulationRunner`
  - Flow
    - Runner iteratively asks both participants for their next message
    - Participants generate messages and a timestamp of the next message
    - Runner selects the next message based on the timestamps
    - Outcome detector checks if the conversation has reached an outcome
  - The conversation ends when each participant has nothing to say, or if the outcome detector detects an outcome (several messages can still be exchanged afterwards)
- **Output**: List of simulated conversations
#### Outcome Detection
- **Input**: Conversation state
- **Process**: 
  - Analyze the conversation for predefined outcomes
  - Use the `OutcomeDetector` to identify if an outcome was reached
- **Output**: outcome if detected, otherwise None
