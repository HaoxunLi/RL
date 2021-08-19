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
actInfo = rlNumericSpec([4 1],'LowerLimit',[0;-0.8;0;-0.8],'UpperLimit',[0;0.8;0;0.8]);
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
%% Create DDPG Agent 

observationPath = [
    featureInputLayer(6,'Normalization','none','Name','observation')
    fullyConnectedLayer(128,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(128,'Name','fc2')
    additionLayer(2,'Name','add')
    reluLayer('Name','relu2')
    fullyConnectedLayer(64,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(1,'Name','fc4')];
actionPath = [
    featureInputLayer(4,'Normalization','none','Name','action')
    fullyConnectedLayer(128,'Name','fc5')];
% Create the layer graph.
criticNetwork = layerGraph(observationPath);
criticNetwork = addLayers(criticNetwork,actionPath);

% Connect actionPath to observationPath.
criticNetwork = connectLayers(criticNetwork,'fc5','add/in2');
criticOptions = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1);
critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,...
    'Observation',{'observation'},'Action',{'action'},criticOptions);
actorNetwork = [
    featureInputLayer(6,'Normalization','none','Name','observation')
    fullyConnectedLayer(128,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(128,'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(64,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(4,'Name','fc4')
    tanhLayer('Name','tanh1')];

actorOptions = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1);

actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,...
    'Observation',{'observation'},'Action',{'tanh1'},actorOptions);
agentOptions = rlDDPGAgentOptions(...
    'SampleTime',Ts,...
    'TargetSmoothFactor',1e-3,...
    'ExperienceBufferLength',1e6 ,...
    'DiscountFactor',0.99,...
    'MiniBatchSize',256);
agentOptions.NoiseOptions.StandardDeviation = 1e-1;
agentOptions.NoiseOptions.StandardDeviationDecayRate = 1e-6;
agent = rlDDPGAgent(actor,critic,agentOptions);


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

