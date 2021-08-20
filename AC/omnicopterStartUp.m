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
vectors = {[-0.1 0.2 -0.1 -0.2]',...
 [-0.1 -0.2 -0.1 0.2]',...
 [-0.1 0.4 -0.1 -0.4]',...
 [-0.1 -0.4 -0.1 0.4]',...
 [-0.1 0.6 -0.1 -0.6]',...
 [-0.1 -0.6 -0.1 0.6]',...
 [-0.1 0.8 -0.1 -0.8]',...
 [-0.1 -0.8 -0.1 0.8]',...
 [0 0.2 0 -0.2]',...
 [0 -0.2 0 0.2]',...
 [0 0.4 0 -0.4]',...
 [0 -0.4 0 0.4]',...
 [0 0.6 0 -0.6]',...
 [0 -0.6 0 0.6]',...
 [0 0.8 0 -0.8]',...
 [0 -0.8 0 0.8]',...
 [0.1 0.2 0.1 -0.2]',...
 [0.1 -0.2 0.1 0.2]',...
 [0.1 0.4 0.1 -0.4]',...
 [0.1 -0.4 0.1 0.4]',...
 [0.1 0.6 0.1 -0.6]',...
 [0.1 -0.6 0.1 0.6]',...
 [0.1 0.8 0.1 -0.8]',...
 [0.1 -0.8 0.1 0.8]'};
actInfo = rlFiniteSetSpec(vectors);
actInfo.Name = 'Thrust;Angular Rate';
actInfo.Description = {'Left Thrust','Left Angular Rate','Right Thrust','Right Angular Rate'};
%% Build Custom Environment
env = rlSimulinkEnv(mdl,agentblk,obsInfo,actInfo);
%% Extract Data from Environment
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
% Specify the simulation time Tf and the agent sample time Ts in seconds.
Ts = 0.1;
Tf = 20;
rng(0)
%% Create AC Agent
criticNetwork = [
    imageInputLayer([6 1 1],'Normalization','none','Name','state')
    fullyConnectedLayer(400,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(200,'Name','CriticStateFC2')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','CriticFC')];
criticOpts = rlRepresentationOptions('LearnRate',1e-3,'GradientThreshold',1);
critic = rlValueRepresentation(criticNetwork,obsInfo,'Observation',{'state'},criticOpts);


actorNetwork = [
    imageInputLayer([6 1 1],'Normalization','none','Name','state')
    fullyConnectedLayer(400,'Name','ActorStateFC1')
    reluLayer('Name','ActorRelu1')
    fullyConnectedLayer(200,'Name','ActorStateFC2')
    reluLayer('Name','ActorRelu2')
    fullyConnectedLayer(24,'Name','action')];

actorOpts = rlRepresentationOptions('LearnRate',1e-3,'GradientThreshold',1);

actor = rlStochasticActorRepresentation(actorNetwork,obsInfo,actInfo,...
    'Observation',{'state'},actorOpts);

agentOpts = rlACAgentOptions(...
    'NumStepsToLookAhead',32, ...
    'DiscountFactor',0.99);

agent = rlACAgent(actor,critic,agentOpts);

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
totalReward = sum(experience.Reward);