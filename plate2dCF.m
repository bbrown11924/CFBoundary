close all;

%problem parameters
D = 1.0e0; %stiffness parameter
Lx = 1; %length in x direction
Ly = 1; %length in y direction
Nx = 10; %number of nodes in x direction
Ny = 10; %number of nodes in y direction
N = (Nx-1)*(Ny); %total number of nodes where w is unknown
nu=0.3;
k0 = 0.1; %damping parameter
k1 = 0; %damping parameter
a1 = 0;
a2 = 0;
anim = 1; %see animation
energies = 1; %compute and plot energies

%timestepping parameters
t0 = 0; %initial time
tf = 10;
0e0; %final time
ns = ceil(50*tf)+1; %number of time steps
t = linspace(t0, tf, ns); %time vector


%initialize spatial domain
x = linspace(0,Lx,Nx);
y = linspace(0,Ly,Ny);
[X,Y] = meshgrid(x,y);
dx = x(2)-x(1);
dy = y(2)-y(1);

%initialize solution array W = [wn; vn];
%get initial diplacement and velocity - form W0
wn = winit(X,Y);
vn = vinit(X,Y);
W0 = [wn; vn]; %initial data vector is of length 2*N


% dydt = RHS(t,W0,D,Nx,Ny,N,x,y,dx,dy,nu);
% dydt = RHS(t,dydt,D,Nx,Ny,N,x,y,dx,dy,nu);
% dydt = RHS(t,dydt,D,Nx,Ny,N,x,y,dx,dy,nu);
% dydt = RHS(t,dydt,D,Nx,Ny,N,x,y,dx,dy,nu);

% call the timestepping code
tic;
[T,w] = ode15s(@(t,W) RHS(t,W,D,Nx,Ny,N,x,y,dx,dy,nu,k0,k1,a1,a2),t,W0);
odetime = toc;
fprintf('Total run time for ODE integrator: %g\n',odetime);

%postprocess the results
%plot the surface
wmat = [zeros(Ny,1) reshape(w(end,1:N),[Nx-1 Ny])'];

% figure(3); surf(X,Y,wmat); %plot the final surface
% zlabel({'w(x,y)'});
% ylabel({'y'});
% xlabel({'x'});
% title('Solution at final time');

%animate the solution
wvals=w(:,1:N);
wmax=max(max(abs(wvals)));
if anim>0
    fig = figure(11);%hold on
    F(ns) = struct('cdata',[],'colormap',[]);
    grid on;
    plt=surf(X,Y,[zeros(Ny,1) reshape(w(1,1:N),[Nx-1 Ny])' ]);
    colormap(jet(256));
    colorbar;
    caxis([-1.1*wmax,1.1*wmax]);
    zlim([-1.1*wmax,1.1*wmax])
    view(-40,30);
    str=sprintf('t = %3.3f',T(1));
    h=text(0.95,0.95,0.9*wmax,str,'FontSize',12);
    F(1) = getframe(fig);
    for j = 2:ns
        plt.ZData =[zeros(Ny,1) reshape(w(j,1:N),[Nx-1 Ny])'];
        str=sprintf('t = %3.3f',T(j));
        set(h,'String',str);
        drawnow % display updates
        F(j) = getframe(fig);
    end
    movie(fig,F,2,anim*ns/tf);
    
    vid = VideoWriter('animation.mp4','MPEG-4');
    open(vid);
    writeVideo(vid,F);
    close(vid);
end

% compute energies if desired
if energies==1
    tic;
    %compute and plot energies
    Le = zeros(size(ns));
    Nle = zeros(size(ns));
    for j=1:ns
        Le(j)= computeEnergies(w(j,:),Nx,Ny,x,y,dx,dy,D,nu);
    end
    energytime = toc;
    fprintf('Total run time for energy calculation: %g\n',energytime);
    figure;hold
    plot(T,Le,'-b', 'DisplayName','E(t)');
    legend('linear E(t)')
end

%end



%function for initial displacement
function w = winit(X,Y)

Lx = X(1,end); %get Lx
Ly = Y(end,1); %get Ly
[Ny,Nx] = size(X); %get Nx and Ny
%wmat = (-4*X.^5+15*X.^4-20*X.^3+10*X.^2); %define the initial value of w
wmat = 0.0*X; %define the initial value of w
% figure; surf(X,Y,wmat); %plot the initial surface
% zlabel({'w(x,y)'});
% ylabel({'y'});
% xlabel({'x'});
% title('Initial Displacement');
wmat = wmat(:,2:Nx); %trim off left values
w = reshape(wmat',[],1); %reshape matrix into the w vector
end

%function for initial velocity
function v = vinit(X,Y)
Lx = X(1,end); %get Lx
Ly = Y(end,1); %get Ly
[Ny,Nx] = size(X); %get Nx and Ny
vmat = X; %define the initial value of w
% figure; surf(X,Y,vmat); %plot the initial surface
% % Create zlabel
% zlabel({'v(x,y)'});
% % Create ylabel
% ylabel({'y'});
% % Create xlabel
% xlabel({'x'});
% title('Initial Velocity');
vmat = vmat(:,2:Nx); %trim off left
v = reshape(vmat',[],1); %reshape matrix into the w vector
end

%function for computing energies
function [le,nle] = computeEnergies(w,Nx,Ny,x,y,dx,dy,D,nu)

%separate the displacements from velocities
w = reshape(w,[Nx-1 2*(Ny)])';
wmat = [zeros(Ny,1),w(1:Ny,:)]; %displacements at interior nodes
vmat = [zeros(Ny,1),w(Ny+1:2*Ny,:)]; %velocities at interior nodes

%compute the appropriate derivatives at the interior points
[lx,ly,lxy] = laplace(wmat,Nx,Ny,dx,dy,nu);
%compute the integral of grad w squared
integrand = nu * (lx + ly).^2 + (1-nu)*(lx.^2 + 2.*lxy.^2 + ly.^2);
normpotential2 = trapz(y,trapz(x,integrand,2));

%compute the integral of the velocities squared
normvel2 = trapz(y,trapz(x,vmat.*vmat,2));

%compute linear energy
le = 0.5*D*normpotential2 + 0.5*normvel2;

end
function [lx,ly,lxy] = laplace(w,Nx,Ny,dx,dy,nu)

%initialize the output arrays
lx = zeros(size(w));
ly = zeros(size(w));
lxy = zeros(size(w));
%apply BCs to wmat so we can compute w_xx and w_yy
%build the array with ghost nodes
w = [zeros(1,size(w,2)+2); [zeros(size(w,1),1), w, zeros(size(w,1),1)]; zeros(1,size(w,2)+2)];

%Clamped edge ghost points
w(:,1) = w(:,3);

%free edge (avoiding free-free corner area)
for j=2:Nx
    i = Ny+1;
    w(i+1,j) = -nu * (w(i,j-1) - 2 * w(i,j) + w(i,j+1)) / dx ^ 2 * dy ^ 2 - w(i-1,j) + 2 * w(i,j);
end

for j=2:Nx
    i = 2;
    w(i-1,j) = -nu * (w(i,j-1) - 2 * w(i,j) + w(i,j+1)) / dx ^ 2 * dy ^ 2 - w(i+1,j) + 2 * w(i,j);
end

for i=3:Ny+1
    j = Nx+1;
    w(i,j+1) = -nu * (w(i-1,j) - 2 * w(i,j) + w(i+1,j)) / dy ^ 2 * dx ^ 2 - w(i,j-1) + 2 * w(i,j);
end


%Now we will apply the conditions for the corners.
%Labelling convention in this section is relative to the corner

%Top right corner
i = 2;
j = Nx + 1;
%E
w(i,j+1) = 2 * w(i,j) - w(i,j-1);

%N
w(i-1,j) = 2 * w(i,j) - w(i+1,j);

%NE
w(i-1,j+1) = w(i+1,j+1) - w(i+1,j-1) + w(i-1,j-1);


%Bottom right corner
i = Ny + 1;
j = Nx + 1;
%E
w(i,j+1) = 2 * w(i,j) - w(i,j-1);


%S
w(i+1,j) = 2 * w(i,j) - w(i-1,j);

%SE
w(i+1,j+1) = w(i-1,j+1) - w(i-1,j-1) + w(i+1,j-1);

%loop over interior nodes and compute w_x and w_y using 2nd order centered
%differences
for j = 2:Nx+1 %which row of nodes
    for i=2:Ny+1 %which column of nodes
        lx(i-1,j-1) = (1/(dx^2))*(w(i,j+1)-2*w(i,j)+w(i,j-1)); %w_xx calculation
        ly(i-1,j-1) = (1/(dy^2))*(w(i+1,j)-2*w(i,j)+w(i-1,j)); %w_yy calculation
        lxy(i-1,j-1) = (1/(4*dx*dy)) * ( w(i+1,j+1) - w(i-1,j+1) - w(i+1,j-1) + w(i-1,j-1));
    end
end
end

function [gx,gy] = grads(w,Nx,Ny,dx,dy)

gx = zeros(size(w));
gy = zeros(size(w));

%apply BCs to wmat so we can compute w_xx and w_yy
%build the array with ghost nodes
w = [zeros(1,size(w,2)+2); [zeros(size(w,1),1), w, zeros(size(w,1),1)]; zeros(1,size(w,2)+2)];
%Applying hinged-hinged conditions
w(1,:) = -1.0 * w(3,:);
w(end,:) = -1.0 * w(end-2,:);

w(:,1) = -1.0 * w(:,3);
w(:,end) = -1.0 * w(:,end-2);
%loop over interior nodes and compute w_x and w_y using 2nd order centered
%differences
for j = 2:Ny+1 %which row of nodes
    for i=2:Nx-1 %which column of nodes
        gx(j-1,i) = (1/(2*dx))*(w(j,i+1)-w(j,i-1)); %w_x calculation
        gy(j-1,i) = (1/(2*dy))*(w(j+1,i)-w(j-1,i)); %w_y calculation
    end
end
end

%define the forcing function f(x,y,t)
function z = f(~,~,~)
z=0;
end

%define the ODE RHS function
function dydt=RHS(t,W,D,Nx,Ny,N,x,y,dx,dy,nu,k0,k1,a1,a2)
fprintf('Time: %g\n',t);
%reshape W into a matrix with 2*Ny rows and Nx-2 columns
wmat = reshape(W,[Nx-1 2*(Ny)])';
w = wmat(1:Ny,:); %displacements at interior nodes
v = wmat(Ny+1:2*Ny,:); %velocities at interior nodes

%initialize dydt
dydt = zeros(size([w;v]));
boundaryWxx = [ones(size(w,1),1) ones(size(w))];
boundaryWxxx = [ones(size(w,1),1) ones(size(w))];

dydt(1:Ny,1:Nx-1) = v; %place v values in the w_t locations

%pad v vector with zeros on all sides
v = [zeros(1,size(v,2)+2); [zeros(size(v,1),1), v, zeros(size(v,1),1)]; zeros(1,size(v,2)+2)];
v = [zeros(1,size(v,2)+2); [zeros(size(v,1),1), v, zeros(size(v,1),1)]; zeros(1,size(v,2)+2)];


gradarr=[zeros(size(w,1),1), w];
[gradx,grady] = grads(gradarr,Nx,Ny,dx,dy);
normgrad2 = trapz(y,trapz(x,gradx.*gradx + grady.*grady,2));

%pad w vector with zeros on all sides to calculate the grad w squared
w = [zeros(1,size(w,2)+2); [zeros(size(w,1),1), w, zeros(size(w,1),1)]; zeros(1,size(w,2)+2)];

w = [zeros(1,size(w,2)+2); [zeros(size(w,1),1), w, zeros(size(w,1),1)]; zeros(1,size(w,2)+2)];

%Clamped edge ghost points
w(:,1) = w(:,3);

%free edge (avoiding free-free corner area)
for j=2:Nx
    i = Ny+2;
    w(i+1,j) = -nu * (w(i,j-1) - 2 * w(i,j) + w(i,j+1)) / dx ^ 2 * dy ^ 2 - w(i-1,j) + 2 * w(i,j);
end

for j=2:Nx
    i = 3;
    w(i-1,j) = -nu * (w(i,j-1) - 2 * w(i,j) + w(i,j+1)) / dx ^ 2 * dy ^ 2 - w(i+1,j) + 2 * w(i,j);
end

for i=4:Ny+1
    j = Nx+1;
    w(i,j+1) = -nu * (w(i-1,j) - 2 * w(i,j) + w(i+1,j)) / dy ^ 2 * dx ^ 2 - w(i,j-1) + 2 * w(i,j);
end

%free edge, 2nd row (avoiding free-free corner area)

for j=3:Nx-1
    i = Ny+2;
    w(i + 2, j) = -(2 - nu) / dx ^ 2 * dy ^ 2 * (w(i + 1, j + 1) - w(i - 1, j + 1) - 2 * w(i + 1, j) + 2 * w(i - 1, j) + w(i + 1, j - 1) - w(i - 1, j - 1)) + 2 * w(i + 1, j) - 2 * w(i - 1, j) + w(i - 2, j);
end

for j=3:Nx-1
    i = 3;
    w(i - 2, j) = (2 - nu) / dx ^ 2 * dy ^ 2 * (w(i + 1, j + 1) - w(i - 1, j + 1) - 2 * w(i + 1, j) + 2 * w(i - 1, j) + w(i + 1, j - 1) - w(i - 1, j - 1)) + w(i + 2, j) - 2 * w(i + 1, j) + 2 * w(i - 1, j);
end

for i=5:Ny
    j = Nx+1;
    w(i,j+2) = -(2 - nu) * (w(i + 1, j + 1) - w(i + 1, j - 1) - 2 * w(i, j + 1) + 2 * w(i, j - 1) + w(i - 1, j + 1) - w(i - 1, j - 1)) / dy ^ 2 * dx ^ 2 + 2 * w(i, j + 1) - 2 * w(i, j - 1) + w(i, j - 2);
end

%Now we will apply the conditions for the corners.
%Labelling convention in this section is relative to the corner

%Top right corner
i = 3;
j = Nx + 1;
%E
w(i,j+1) = 2 * w(i,j) - w(i,j-1);

%EE
w(i, j + 2) = (-4 * (nu - 2) * (w(i, j) - w(i, j - 1) - w(i + 1, j + 1) / 2 + w(i + 1, j - 1) / 2) * dx ^ 2 + 4 * (w(i, j) - w(i, j - 1) + w(i, j - 2) / 4) * dy ^ 2) / dy ^ 2;

%N
w(i-1,j) = 2 * w(i,j) - w(i+1,j);

%NE
w(i-1,j+1) = w(i+1,j+1) - w(i+1,j-1) + w(i-1,j-1);

%NN
w(i-2, j) = (-4 * (nu - 2) * (w(i, j) - w(i + 1, j) - w(i - 1, j - 1) / 2 + w(i + 1, j - 1) / 2) * dy ^ 2 + 4 * dx ^ 2 * (w(i, j) - w(i + 1, j) + w(i + 2, j) / 4)) / dx ^ 2;

%NNW
w(i-2, j-1) = (2 * (w(i, j) - w(i + 1, j - 2) / 2 - w(i + 1, j) - w(i - 1, j - 1) + w(i + 1, j - 1) + w(i - 1, j - 2) / 2) * (nu - 2) * dy ^ 2 + 2 * dx ^ 2 * (w(i - 1, j - 1) - w(i + 1, j - 1) + w(i + 2, j - 1) / 2)) / dx ^ 2;

%SEE
w(i+1, j+2) = (2 * (w(i, j) - w(i, j - 1) - w(i + 1, j + 1) + w(i + 1, j - 1) + w(i + 2, j + 1) / 2 - w(i + 2, j - 1) / 2) * (nu - 2) * dx ^ 2 + dy ^ 2 * (w(i + 1, j - 2) + 2 * w(i + 1, j + 1) - 2 * w(i + 1, j - 1))) / dy ^ 2;


%Bottom right corner
i = Ny + 2;
j = Nx + 1;
%E
w(i,j+1) = 2 * w(i,j) - w(i,j-1);

%EE
w(i, j + 2) = (-4 * (nu - 2) * (w(i, j) - w(i, j - 1) - w(i - 1, j + 1) / 2 + w(i - 1, j - 1) / 2) * dx ^ 2 + 4 * (w(i, j) - w(i, j - 1) + w(i, j - 2) / 4) * dy ^ 2) / dy ^ 2;

%S
w(i+1,j) = 2 * w(i,j) - w(i-1,j);

%SE
w(i+1,j+1) = w(i-1,j+1) - w(i-1,j-1) + w(i+1,j-1);

%SS
w(i+2, j) = (-4 * (nu - 2) * (w(i, j) - w(i - 1, j) - w(i + 1, j - 1) / 2 + w(i - 1, j - 1) / 2) * dy ^ 2 + 4 * dx ^ 2 * (w(i, j) - w(i - 1, j) + w(i- 2, j) / 4)) / dx ^ 2;

%SSW
w(i+2, j-1) = (2 * (w(i, j) - w(i - 1, j - 2) / 2 - w(i - 1, j) - w(i + 1, j - 1) + w(i - 1, j - 1) + w(i + 1, j - 2) / 2) * (nu - 2) * dy ^ 2 + 2 * dx ^ 2 * (w(i + 1, j - 1) - w(i - 1, j - 1) + w(i - 2, j - 1) / 2)) / dx ^ 2;

%NEE
w(i-1, j+2) = (2 * (w(i, j) - w(i, j - 1) - w(i - 1, j + 1) + w(i - 1, j - 1) + w(i - 2, j + 1) / 2 - w(i - 2, j - 1) / 2) * (nu - 2) * dx ^ 2 + dy ^ 2 * (w(i - 1, j - 2) + 2 * w(i - 1, j + 1) - 2 * w(i - 1, j - 1))) / dy ^ 2;



%compute the RHS of the velocity equation
for j = 3:Nx+1 %which column of nodes
    for i=3:Ny+2 %which row of nodes

        %In vacuo conditions
        wxxxx = (1/dx^4) * (w(i-2,j) - 4*w(i-1,j) + 6*w(i,j)...
                - 4 *w(i+1,j) + w(i+2,j));
        
        wyyyy = (1/dy^4) * (w(i,j-2) - 4*w(i,j-1) + 6*w(i,j)...
                - 4 *w(i,j+1) + w(i,j+2));
        
        wxxyy = (1/(dx^2*dy^2)) * ( w(i+1,j+1) - 2* w(i+1,j)...
                + w(i+1,j-1) - 2* w(i,j+1) + 4 * w(i,j) -2*w(i,j-1)...
                +w(i-1,j+1) -2 * w(i-1,j)+w(i-1,j-1));
            
        %damping term
        wt = v(i,j);
        
        wtxx = (1/dx^2) * (v(i+1,j) - 2* v(i,j) + v(i-1,j));
        
        wtyy = (1/dy^2) * (v(i,j+1) - 2* v(i,j) + v(i,j-1));
        
        damping = - k0*(wt) + k1*(wtxx + wtyy);
        
        %spatial
        wx = (1/(2*dx))*(w(i,j+1)-w(i,j-1)); %w_x calculation
        wy = (1/(2*dy))*(w(i+1,j)-w(i-1,j)); %w_y calculation
        
        spatial = - a1 * wx - a2 * wy;
        
         dydt(Ny+(i-2),j-2) =  - 1 * D *(wxxxx + 2 * wxxyy + wyyyy)...
                                 + f(x(j-2),y(i-2),t)...
                                 + spatial + damping;  
    end
end
%dydt
dydt = reshape(dydt',[2*N 1]);

end
