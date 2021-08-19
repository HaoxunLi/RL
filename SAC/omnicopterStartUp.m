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
%% Create SAC Agent 
statePath1 = [
    sequenceInputLayer(6,'Normalization','none','Name','observation')
    fullyConnectedLayer(400,'Name','CriticStateFC1')
    reluLayer('Name','CriticStateRelu1')
    fullyConnectedLayer(300,'Name','CriticStateFC2')
    ];
actionPath1 = [
    sequenceInputLayer(4,'Normalization','none','Name','action')
    fullyConnectedLayer(300,'Name','CriticActionFC1')
    ];
commonPath1 = [
    additionLayer(2,'Name','add')
    lstmLayer(8,'OutputMode','sequence','Name','lstm')
    reluLayer('Name','CriticCommonRelu1')
    fullyConnectedLayer(1,'Name','CriticOutput')
    ];

criticNet = layerGraph(statePath1);
criticNet = addLayers(criticNet,actionPath1);
criticNet = addLayers(criticNet,commonPath1);
criticNet = connectLayers(criticNet,'CriticStateFC2','add/in1');
criticNet = connectLayers(criticNet,'CriticActionFC1','add/in2');
criticOptions = rlRepresentationOptions('Optimizer','adam','LearnRate',1e-3,... 
                                        'GradientThreshold',1,'L2RegularizationFactor',2e-4);
critic1 = rlQValueRepresentation(criticNet,obsInfo,actInfo,...
    'Observation',{'observation'},'Action',{'action'},criticOptions);
critic2 = rlQValueRepresentation(criticNet,obsInfo,actInfo,...
    'Observation',{'observation'},'Action',{'action'},criticOptions);
statePath = [
    sequenceInputLayer(6,'Normalization','none','Name','observation')
    fullyConnectedLayer(400, 'Name','commonFC1')
    lstmLayer(8,'OutputMode','sequence','Name','lstm')    
    reluLayer('Name','CommonRelu')];
meanPath = [
    fullyConnectedLayer(300,'Name','MeanFC1')
    reluLayer('Name','MeanRelu')
    fullyConnectedLayer(4,'Name','Mean')
    ];
stdPath = [
    fullyConnectedLayer(300,'Name','StdFC1')
    reluLayer('Name','StdRelu')
    fullyConnectedLayer(4,'Name','StdFC2')
    softplusLayer('Name','StandardDeviation')];

concatPath = concatenationLayer(1,2,'Name','GaussianParameters');

actorNetwork = layerGraph(statePath);
actorNetwork = addLayers(actorNetwork,meanPath);
actorNetwork = addLayers(actorNetwork,stdPath);
actorNetwork = addLayers(actorNetwork,concatPath);
actorNetwork = connectLayers(actorNetwork,'CommonRelu','MeanFC1/in');
actorNetwork = connectLayers(actorNetwork,'CommonRelu','StdFC1/in');
actorNetwork = connectLayers(actorNetwork,'Mean','GaussianParameters/in1');
actorNetwork = connectLayers(actorNetwork,'StandardDeviation','GaussianParameters/in2');
actorOptions = rlRepresentationOptions('Optimizer','adam','LearnRate',1e-3,...
                                       'GradientThreshold',1,'L2RegularizationFactor',1e-5);

actor = rlStochasticActorRepresentation(actorNetwork,obsInfo,actInfo,actorOptions,...
    'Observation',{'observation'});
agentOptions = rlSACAgentOptions;
agentOptions.SampleTime = Ts;
agentOptions.DiscountFactor = 0.99;
agentOptions.TargetSmoothFactor = 1e-3;
agentOptions.ExperienceBufferLength = 1e6;
agentOptions.SequenceLength = 32;
agentOptions.MiniBatchSize = 32;
agent = rlSACAgent(actor,[critic1 critic2],agentOptions);
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

