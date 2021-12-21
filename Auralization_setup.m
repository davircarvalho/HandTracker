clear all;  clc
% DAVI ROCHA CARVALHO APRIL/2021 - Eng. Acustica @UFSM 
% Test binaural rendering using webcam head tracker 

% MATLAB R2021b
%% Carregar HRTFs
ARIDataset = load('ReferenceHRTF.mat'); 
% separar HRTFs
hrtfData = double(ARIDataset.hrtfData);
hrtfData = permute(hrtfData,[2,3,1]);
% Separar posições de fonte
sourcePosition = ARIDataset.sourcePosition(:,[1,2]);
sourcePosition(:,1) = -(sourcePosition(:,1) - 180);


%% Carregar audio mono
[heli,originalSampleRate] = audioread('Heli_16ch_ACN_SN3D.wav'); % ja vem com o matlab 
heli = 12*heli(:,1); % keep only one channel

sampleRate = 48e3;
heli = resample(heli,sampleRate,originalSampleRate);


%% Jogar audio para objeto DSP
sigsrc = dsp.SignalSource(heli, ...
    'SamplesPerFrame',512, ...
    'SignalEndAction','Cyclic repetition');

% Configurar dispositivo de audio
deviceWriter = audioDeviceWriter('SampleRate',sampleRate, ...
    'ChannelMapping',[1,2]);


%% Definir filtros FIR 
FIR = cell(1,2);
FIR{1} = dsp.FIRFilter('NumeratorSource','Input port');
FIR{2} = dsp.FIRFilter('NumeratorSource','Input port');


%% Inicializar Head Tracker 
% open('HeadTracker.exe')
udpr = dsp.UDPReceiver('RemoteIPAddress', '127.0.0.1',...
                       'LocalIPPort',50060, ...
                       'ReceiveBufferSize', 18); % conectar matlab ao head tracker

%% Processamento em tempo real (fonte fixa no espaco)
audioUnderruns = 0;
audioFiltered = zeros(sigsrc.SamplesPerFrame,2);

yaw = 0;
pitch = 0;
hand_state = 0;

s_azim = 0;
s_elev = 0;
msrd_dist = 0.8;

idx_pos = dsearchn(sourcePosition, [s_azim, s_elev]);
HRIR = squeeze((hrtfData(idx_pos, :,:))); 


release(deviceWriter)
release(sigsrc)
tic % start head tracker extrapolation time estimate
t2 = 0;

while true
    tic
    % Ler orientação atual do HeadTracker.
    py_output = step(udpr);
    
    if ~isempty(py_output)
        data = str2double(split(convertCharsToStrings(char(py_output)), ','));
        head_yaw = data(1);
        head_pitch = data(2);
        head_roll = data(3);
        hand_state = data(4);
        hand_azimuth = data(5);
        hand_elevation = data(6);
        hand_radius = data(7);
    end
    
    if hand_state
        idx_pos = dsearchn(sourcePosition, [hand_azimuth, 0]);
%         disp(['Azimuth: ' num2str(sourcePosition(idx_pos,1)), '  closed'])
        % Obtain a pair of HRTFs at the desired position.
        HRIR = squeeze((hrtfData(idx_pos, :,:))); 
        % Calculate distance normalization                   
        DistNorm = msrd_dist/hand_radius;     
        HRIR = HRIR .* DistNorm;
    else
%         disp(['Azimuth: ' num2str(sourcePosition(idx_pos,1)), '  open'])
    end     
    

    % Read audio from file   
    audioIn = sigsrc();
             
    % Apply HRTFs
    audioFiltered(:,1) = FIR{1}(audioIn, HRIR(1,:)); % Left
    audioFiltered(:,2) = FIR{2}(audioIn, HRIR(2,:)); % Right    
%     deviceWriter(squeeze(audioFiltered)); 
end
release(sigsrc)
release(deviceWriter)


