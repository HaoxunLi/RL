function plotOmnicopter(y,z,phi,Tl,phil,Tr,phir,Lim)

persistent f

sc = 4;
xb = sc*[0 0.2 0.2 0.15 0.15 -0.15 -0.15 -0.2 -0.2];
yb = sc*[0 0 0.05 0.05 0.025 0.025 0.05 0.05 0];
xT = sc*[-0.01 0.01 0.01 -0.01];
yT = 0.25*sc*[0 0 0.2 0.2];
xp = sc*[-0.1 0.1 0.1 -0.1];
yp = sc*[0 0 -0.01 -0.01];
offl = sc*[-0.175;0.025;0];
offr = sc*[0.175;0.025;0];
offp = [0;1];

if isempty(f) || ~isvalid(f)
    f = figure(...
        'Toolbar','none',...
        'NumberTitle','off',...
        'Name','Omnicopter Visualisation',...
        'Visible','on',...
        'MenuBar','none');
    
    ha = gca(f);
    localResetAxes(ha,Lim)
    
    grid(ha,'on');
    hold(ha,'on');
end

ha = gca(f);
axis(Lim);

Cb = [cos(phi) sin(phi) y;...
     -sin(phi) cos(phi) -z;...
       0 0 1];
CTl = [cos(phil) sin(phil) offl(1,1);...
      -sin(phil) cos(phil) offl(2,1);...
        0 0 1];
CTr = [cos(phir) sin(phir) offr(1,1);...
      -sin(phir) cos(phir) offr(2,1);...
        0 0 1];
Mb = [xb;yb;ones(1,length(xb))];
Mbnew = zeros(size(Mb));
for i=1:length(xb)
    Mbnew(:,i) = Cb*Mb(1:3,i);
end

MTl = [xT;yT*(5*Tl+5);ones(1,length(xT))];
MTr = [xT;yT*(5*Tr+5);ones(1,length(xT))];
MTlnew = zeros(size(MTl));
MTrnew = zeros(size(MTr));
for i=1:length(xT)
    MTlnew(:,i) = Cb*(CTl*MTl(1:3,i));
    MTrnew(:,i) = Cb*(CTr*MTr(1:3,i));
end

body = findobj(ha,'Tag','body');
platform = findobj(ha,'Tag','platform');
leftR = findobj(ha,'Tag','leftR');
rightR = findobj(ha,'Tag','rightR');

if isempty(platform)
    patch(xp+offp(1,1),yp+offp(2,1) ,'g','Tag','platform');
end
if isempty(body)
    patch(xb,yb,'b','Tag','body');
else
    body.XData = Mbnew(1,:);
    body.YData = Mbnew(2,:);
end
if isempty(leftR)
    patch(xT+offl(1,1),yT+offl(2,1),'r','Tag','leftR');
else
    leftR.XData = MTlnew(1,:);
    leftR.YData = MTlnew(2,:);
end
if isempty(rightR)
    patch(xT+offr(1,1),yT+offr(2,1),'r','Tag','rightR');
else
    rightR.XData = MTrnew(1,:);
    rightR.YData = MTrnew(2,:);
end

drawnow();

end

function localResetAxes(ha,Lim)
cla(ha);
set(ha,'XLim',[Lim(1) Lim(2)]);
set(ha,'YLim',[Lim(3),Lim(4)]);
axis equal;
end

