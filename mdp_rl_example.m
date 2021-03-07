clc
clear all
close all

%MDP Example for reinforcement learning

% Creating an MDP environment

% MDP = createMDP(6,["accept";"reject"]);
MDP = createMDP(8,["up";"down"]);

% State 1 Transition and Reward
MDP.T(1,2,1) = 1;
MDP.R(1,2,1) = 3;
MDP.T(1,3,2) = 1;
MDP.R(1,3,2) = 1;

% State 2 Transition and Reward
MDP.T(2,4,1) = 1;
MDP.R(2,4,1) = 2;
MDP.T(2,5,2) = 1;
MDP.R(2,5,2) = 1;

% State 3 Transition and Reward
MDP.T(3,5,1) = 1;
MDP.R(3,5,1) = 2;
MDP.T(3,6,2) = 1;
MDP.R(3,6,2) = 4;

% State 4 Transition and Reward
MDP.T(4,7,1) = 1;
MDP.R(4,7,1) = 3;
MDP.T(4,8,2) = 1;
MDP.R(4,8,2) = 2;

% State 5 Transition and Reward
MDP.T(5,7,1) = 1;
MDP.R(5,7,1) = 1;
MDP.T(5,8,2) = 1;
MDP.R(5,8,2) = 9;

% State 6 Transition and Reward
MDP.T(6,7,1) = 1;
MDP.R(6,7,1) = 5;
MDP.T(6,8,2) = 1;
MDP.R(6,8,2) = 1;

% State 7 Transition and Reward
MDP.T(7,7,1) = 1;
MDP.R(7,7,1) = 0;
MDP.T(7,7,2) = 1;
MDP.R(7,7,2) = 0;

% State 8 Transition and Reward
MDP.T(8,8,1) = 1;
MDP.R(8,8,1) = 0;
MDP.T(8,8,2) = 1;
MDP.R(8,8,2) = 0;

%Specify the terminal states of the model
MDP.TerminalStates = ["s7";"s8"];

%MDP.TerminalStates = ["s5";"s6"];  %

env = rlMDPEnv(MDP);


env.ResetFcn = @() 1;


rng(0)

%Creating Q Learning Agent

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
qTable = rlTable(obsInfo, actInfo);
qRepresentation = rlQValueRepresentation(qTable, obsInfo, actInfo);
qRepresentation.Options.LearnRate = 1;



agentOpts = rlQAgentOptions;
agentOpts.DiscountFactor = 1;
agentOpts.EpsilonGreedyExploration.Epsilon = 0.9;
agentOpts.EpsilonGreedyExploration.EpsilonDecay = 0.01;
qAgent = rlQAgent(qRepresentation,agentOpts);


% Train Q Learning Agent

trainOpts = rlTrainingOptions;
trainOpts.MaxStepsPerEpisode = 50;
trainOpts.MaxEpisodes = 300;
trainOpts.StopTrainingCriteria = "AverageReward";
trainOpts.StopTrainingValue = 12;
trainOpts.ScoreAveragingWindowLength = 30;


    trainingStats = train(qAgent,env,trainOpts);
% 
% doTraining = false;
% 
% if doTraining
%     % Train the agent.
%     trainingStats = train(qAgent,env,trainOpts);
% else
%     % Load pretrained agent for the example.
%     %load('genericMDPQAgent.mat','qAgent');
% end

% Validate Q Learning Results

Data = sim(qAgent,env);
cumulativeReward = sum(Data.Reward)

QTable = getLearnableParameters(getCritic(qAgent));
QTable{1}

TrueTableValues = [13,12;5,10;11,9;3,2;1,9;5,1;0,0;0,0]