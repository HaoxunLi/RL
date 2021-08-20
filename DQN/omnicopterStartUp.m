%% Simple omnicopter RL_Example
%
%
%
clear; clc;
%% Environment
mdl = 'omnicopterTest';
open_system(mdl);
agentblk = [mdl '/RL Agent'];
%% Observations
obsInfo = rlNumericSpec([6 1]);
obsInfo.Name = 'Omnicopter Position'; 
obsInfo.Description = {'y,yd,z,zd,phi,phid'};
%% Actions
vectors = {[0 0.2 0 -0.2]',...
 [0 -0.2 0 0.2]',...
 [0 0.4 0 -0.4]',...
 [0 -0.4 0 0.4]',...
 [0 0.6 0 -0.6]',...
 [0 -0.6 0 0.6]',...
 [0 0.8 0 -0.8]',...
 [0 -0.8 0 0.8]'};
actInfo = rlFiniteSetSpec(vectors);
actInfo.Name = 'Thrust;Angular Rate';
actInfo.Description = {'Left Thrust','Left Angular Rate','Right Thrust','Right Angular Rate'};
%% Build Custom Environment
env = rlSimulinkEnv(mdl,agentblk,obsInfo,actInfo)
%% Extract Data from Environment
obsInfo = getObservationInfo(env)
actInfo = getActionInfo(env);
% Specify the simulation time Tf and the agent sample time Ts in seconds.
Ts = 0.1;
Tf = 20;
rng(0)
%% Create DQN Agent
dnn = [
    featureInputLayer(6,'Normalization','none','Name','state')
    fullyConnectedLayer(48,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(96,'Name','CriticStateFC2')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(8,'Name','output')];
criticOpts = rlRepresentationOptions('LearnRate',0.001,'GradientThreshold',1);
critic = rlQValueRepresentation(dnn,obsInfo,actInfo,'Observation',{'state'},'Action',{'output'},criticOpts);
figure
plot(layerGraph(dnn))
agentOpts = rlDQNAgentOptions(...
    'SampleTime',0.05,...
    'TargetSmoothFactor',1e-3,...
    'ExperienceBufferLength',3000,... 
    'UseDoubleDQN',false,...
    'DiscountFactor',0.9,...
    'MiniBatchSize',64);
agent = rlDQNAgent(critic,agentOpts);

%% Parameters
m = 1;              % mass(kg)
Ixx = 0.1;          % roll inertia (kgm^2)
l = 0.2;            % moment arm
eta = 0;            % magnitude of termination error
ymax=5; ymin=-5;    % max and min y-values for environment
zmax=0; zmin=-10;   % max and min z-values for environment
yp = 0; zp = -1;    % location of the landing pad (m)
phip = 0;           % orientation of landing pad (rad)
g = 10;             % acceleration due to gravity
C = [1 0 0 0 0 0;...
     0 0 1 0 0 0;...
     0 0 0 0 1 0];  % pose selection matrix
y = 0;
z = -5;
phi = 0;
actions = [0 0 0 0]';

%% Training Parameters
maxepisodes = 2000;
maxsteps = ceil(Tf/Ts);
trainingOptions = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes,...
    'MaxStepsPerEpisode',maxsteps,...
    'StopOnError',"on",...
    'Verbose',false,...
    'Plots',"training-progress",...
    'StopTrainingCriteria',"AverageReward",...
    'StopTrainingValue',400,...
    'ScoreAveragingWindowLength',10,...
    'SaveAgentCriteria',"EpisodeReward",...
    'SaveAgentValue',400); 

trainingStats = train(agent,env,trainingOptions);
simOptions = rlSimulationOptions('MaxSteps',maxsteps);
experience = sim(env,agent,simOptions);
