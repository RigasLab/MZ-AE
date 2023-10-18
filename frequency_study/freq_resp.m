clear all;
load('test_freq.mat', 'input', 'output', 'time');

output_signal = double(squeeze(output));
input_signal  = double(squeeze(input));

%%
ts = 0.25;
num_sensors = size(output_signal,3);
T = 10; % Total sampling time
num_samp = floor(T/ts); % Total number of samples
fre      = logspace(log10(1/T),log10(1/ts/2),120); % Frequencies used in frequency sweeping

%%
for k = 1:num_sensors %lop over sensor
    for j = 1:120 %loop over frequency
        Out{j}=output_signal(j,:,k)';
        In{j}=input_signal(j,:,k)';
    end
    data  = iddata(Out, In, ts);
    Gw(k) = spafdr(data,[],fre);
    Response = Gw(k).ResponseData;
    sys(k) = frd(Response,fre,ts);
    figure(1)
    [sv(:,:,k),svout] = sigma(sys(k));
         sigma(sys(k));
         hold on
         figure(2);
         bode(sys(k));
         hold on
end
%% SVD for frequency response function matrix
Res = Gw.ResponseData;
SV=zeros(1,size(fre,2));
for i = 1:size(fre,2)
    H = Res(:,:,i);
    [U,S,V] = svd(H);
    SV(i)=S(1);
end

figure(3)
semilogx(fre,20*log10(SV),'-o');

ylabel('Singular Values of Response Matrix','interpreter','latex','Rotation',90);
xlabel('Frequency','interpreter','latex');
% figure(4)
% semilogx(fre,SV,'-o');
% ylabel('SV','interpreter','latex','Rotation',0);
% xlabel('St','interpreter','latex');
% xline(fvs,'LineStyle','--')
% legend("PM-Dynamic","Vortex Shedding",'interpreter','latex')
% %legend('FM-NoDelay','PM-NoDelay','PM-Delay','interpreter','latex')
% set(gca,'TickLabelInterpreter','latex');
% set(gca,'Linewidth',1);
% set(gca,'Fontsize',11);
% %set(gca,'XTick',[0:300:1500])
% hold on

%% PSD of inputs and outputs (Use pburg, pcov, periodogram, pmcov, pmtm, pwelch, or pyulear)

% Inputs
x = zeros(num_samp,1);
fs=1/ts;
for i=1:204
    sensor = i; % Select the number of sensor to calculate
    for j = 1:120
        x(:,1) = Inputs{1,j}(:,sensor);
        y(:,1) = Outputs{1,j}(:,sensor);
        Nx = length(x);
        Ny = length(y);
        % FFT
        xdft = fft(x);
        xdft = xdft(1:Nx/2+1); % Half spectrum
        ydft = fft(y);
        ydft = ydft(1:Ny/2+1);

        % PSD
        psdx = (1/(fs*Nx)) * abs(xdft).^2;
        psdx(2:end-1) = 2*psdx(2:end-1);
        psdy = (1/(fs*Ny)) * abs(ydft).^2;
        psdy(2:end-1) = 2*psdy(2:end-1);
        %     figure(5)
        %     periodogram(x,[],Nx)
        %     hold on

        % Matrix for PSD contour
        PSD(:,j) = pow2db(psdy(2:end));

        freq = 0:fs/Nx:fs/2;
%         figure(1)
%         plot(1:num_samp,y)%,1:num_samp,y)
%         hold on
        %     figure(4)
        %     plot(freq,pow2db(psdx))
        %     grid on
        %     title("Periodogram Using FFT",'interpreter','latex')
        %     xlabel("St",'interpreter','latex')
        %     ylabel("PSD (dB)",'interpreter','latex')
        %     hold on
        %     figure(5)
        %     plot(freq,pow2db(psdy))
        %     grid on
        %     title("PSD-Output (FM-Static)",'interpreter','latex')
        %     xlabel("St",'interpreter','latex')
        %     ylabel("PSD (dB)",'interpreter','latex')
        %     hold on
    end

    if i <= 64
        figure(6)
        subplot(8,8,i)
        contourf(fre,freq(2:end),PSD)
        %xlabel("Input St",'interpreter','latex')
        %ylabel("Output St",'interpreter','latex')
        %clim([-120, -20])
        %     xline(fvs,'LineStyle','--')
        %title("Periodogram Using FFT",'interpreter','latex')
        %legend("PSD",'interpreter','latex')
    elseif i <= 128
        figure(8)
        subplot(8,8,i-64)
        contourf(fre,freq(2:end),PSD) 

    elseif i <= 192
        figure(9)
        subplot(8,8,i-128)
        contourf(fre,freq(2:end),PSD) 

    end
end


