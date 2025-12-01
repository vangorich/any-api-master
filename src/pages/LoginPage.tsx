import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import apiClient from '@/utils/api';
import { useSystem } from '@/contexts/SystemContext';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useToast } from '@/hooks/useToast';
import TurnstileWidget from '@/components/TurnstileWidget';
import { TurnstileInstance } from '@marsidev/react-turnstile';
import { Mail, Lock, User, Key } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import CaptchaDialog from '@/components/CaptchaDialog';

export default function LoginPage() {
    const [isLogin, setIsLogin] = useState(true);
    const [isResetPassword, setIsResetPassword] = useState(false);
    const [username, setUsername] = useState('');
    const [email, setEmail] = useState('');
    useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [verificationCode, setVerificationCode] = useState('');
    const turnstileTokenRef = useRef<string>('');
    useState(''); // 仍然需要，但由弹窗管理
    useState(''); // 同上
    const { config: systemConfig, loading: configLoading } = useSystem();
    const [isCaptchaDialogOpen, setIsCaptchaDialogOpen] = useState(false);
    const [captchaInfoForTurnstile, setCaptchaInfoForTurnstile] = useState<{ captchaId: string, captchaCode: string } | null>(null);
    const [countdown, setCountdown] = useState(0);
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();
    const { toast } = useToast();
    const auth = useAuth();
    const emailInputRef = useRef<HTMLInputElement>(null);
    const turnstileRef = useRef<TurnstileInstance>(null);
    const [currentAction, setCurrentAction] = useState<'login' | 'register' | 'reset' | 'sendCode' | null>(null);

    const validateEmail = (email: string) => {
        if (!email) return "邮箱不能为空";

        const parts = email.split('@');
        if (parts.length !== 2) return "邮箱必须包含一个 @ 符号";

        const localPart = parts[0];
        const domainPart = parts[1];

        if (!/^[a-zA-Z0-9]+$/.test(localPart)) return "@ 符号前的部分只能包含字母和数字";

        const domainParts = domainPart.split('.');
        if (domainParts.length < 2) return "邮箱域名必须包含 . 符号";

        const domainName = domainParts[0];
        const topLevelDomain = domainParts.slice(1).join('.');

        if (!/^[a-zA-Z]+$/.test(domainName)) return "@ 和 . 符号之间的部分只能包含字母";
        if (!/^[a-zA-Z.]+$/.test(topLevelDomain)) return ". 符号后的部分只能包含字母";

        if (systemConfig?.email_whitelist_enabled) {
            const allowedDomains = systemConfig.email_whitelist || [];
            if (!allowedDomains.includes(domainPart)) {
                const allowedDomainsStr = allowedDomains.join(', ');
                return `该邮箱后缀不允许注册。请使用以下后缀的邮箱: ${allowedDomainsStr}`;
            }
        }

        return ""; // 验证通过
    };

    // 依赖于 useSystem hook 的副作用
    useEffect(() => {
        if (systemConfig) {
            // 不再需要在这里预加载验证码
            // 如果密码登录关闭，且注册也关闭，则自动切换到重置密码（如果开启）
            if (!systemConfig.allow_password_login && !systemConfig.allow_registration && systemConfig.require_email_verification) {
                setIsLogin(false);
                setIsResetPassword(true);
            }
        }
    }, [systemConfig]);

    // 发送验证码
    const proceedWithSendCode = async (captchaInfo: { captchaId: string, captchaCode: string } | null, token?: string) => {
        // 最终防线：在发送请求的函数本身进行检查，确保万无一失
        if (!systemConfig?.require_email_verification) {
            // 此处不需要toast，因为上游的handleSendCode已经提示过了
            // 如果是自动触发，静默失败即可
            return;
        }

        // 立即启动倒计时
        setCountdown(60);
        const timer = setInterval(() => {
            setCountdown((prev) => {
                if (prev <= 1) {
                    clearInterval(timer);
                    return 0;
                }
                return prev - 1;
            });
        }, 1000);

        try {
            const codeType = isResetPassword ? 'reset_password' : 'register';
            const emailToSend = isResetPassword ? email : email; // Both now use the 'email' state

            const requestData: any = {
                email: emailToSend,
                type: codeType,
            };

            if (systemConfig?.enable_turnstile && token) {
                requestData.turnstile_token = token;
            }

            if (systemConfig?.enable_captcha) {
                if (captchaInfo) {
                    requestData.captcha_id = captchaInfo.captchaId;
                    requestData.captcha_code = captchaInfo.captchaCode;
                }
            }

            await apiClient.post(`/auth/send-code`, requestData);

            toast({
                variant: 'success',
                title: '验证码已发送',
                description: '请查收邮件，验证码10分钟内有效',
            });
            // 成功后也刷新验证码
            // 验证码在弹窗中处理，这里不需要刷新
        } catch (error: any) {
            toast({
                variant: 'error',
                title: '发送失败',
                description: error.response?.data?.detail || '无法发送验证码',
            });
            setCountdown(0); // 发送失败时重置倒计时
        } finally {
            turnstileRef.current?.reset();
        }
    };

    // 新的发送验证码处理流程
    const handleSendCode = async () => {
        // 前置检查：如果系统根本不需要邮箱验证，则直接阻止
        if (!systemConfig?.require_email_verification) {
            toast({
                variant: 'error',
                title: '操作无效',
                description: '系统当前未开启邮箱验证功能。',
            });
            return;
        }

        const isEmailInputValid = isResetPassword
            ? !!email
            : emailInputRef.current?.checkValidity();

        if (!isEmailInputValid) {
            if (!isResetPassword) emailInputRef.current?.reportValidity();
            return;
        }
        
        // 如果开启了图形验证码，则打开弹窗
        if (systemConfig?.enable_captcha) {
            setIsCaptchaDialogOpen(true);
        } else {
            // 否则，直接进入下一步（Turnstile或直接发送）
            triggerSendCodeAction();
        }
    };

    // 当 Captcha 弹窗验证成功后调用此函数
    const onCaptchaSuccess = (captchaId: string, captchaCode: string) => {
        const captchaInfo = { captchaId, captchaCode };
        // 如果需要 Turnstile，则先保存验证码信息，再触发 Turnstile
        if (systemConfig?.enable_turnstile && systemConfig.turnstile_site_key) {
            setCaptchaInfoForTurnstile(captchaInfo);
            setCurrentAction('sendCode');
            turnstileRef.current?.execute();
        } else {
            // 否则直接发送
            proceedWithSendCode(captchaInfo);
        }
    };

    // 当 Turnstile 验证成功后调用此函数
    const handleTurnstileVerify = (token: string) => {
        // 关键修复：在这里也添加检查，防止绕过 handleSendCode 的直接调用
        if (!systemConfig?.require_email_verification) {
            // 在这里可以选择静默失败或给出提示
            // 为避免干扰用户，选择静默处理并重置
            turnstileRef.current?.reset();
            return;
        }
        turnstileTokenRef.current = token;
        
        switch (currentAction) {
            case 'login':
                handleLogin(new Event('submit') as unknown as React.FormEvent);
                break;
            case 'register':
                handleRegister(new Event('submit') as unknown as React.FormEvent);
                break;
            case 'reset':
                handleResetPassword(new Event('submit') as unknown as React.FormEvent);
                break;
            case 'sendCode':
                proceedWithSendCode(captchaInfoForTurnstile, token);
                break;
        }

        // 重置
        setCaptchaInfoForTurnstile(null);
        setCurrentAction(null);
        turnstileRef.current?.reset();
    };

    // 触发发送邮件的最终逻辑 (仅用于无验证码的场景)
    const triggerSendCodeAction = () => {
        if (systemConfig?.enable_turnstile && systemConfig.turnstile_site_key) {
            setCaptchaInfoForTurnstile(null); // 确保没有旧的验证码信息
            setCurrentAction('sendCode');
            turnstileRef.current?.execute();
        } else {
            proceedWithSendCode(null);
        }
    };

    // 验证验证码
    const verifyCode = async () => {
        try {
            await apiClient.post(`/auth/verify-code`, {
                email,
                code: verificationCode,
                type: 'register'
            });
            return true;
        } catch (error: any) {
            toast({
                variant: 'error',
                title: '验证失败',
                description: error.response?.data?.detail || '验证码无效或已过期',
            });
            return false;
        }
    };

    // 登录
    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!systemConfig?.allow_password_login) {
            toast({
                variant: 'error',
                title: '登录已禁用',
                description: '管理员已关闭密码登录功能。',
            });
            return;
        }

        // Turnstile 流程
        if (systemConfig?.enable_turnstile && !turnstileTokenRef.current) {
            setCurrentAction('login');
            turnstileRef.current?.execute();
            return; // 等待 onVerify 回调
        }

        setLoading(true);

        try {
            const formData = new FormData();
            formData.append('username', username);
            formData.append('password', password);

            if (systemConfig?.enable_turnstile) {
                formData.append('turnstile_token', turnstileTokenRef.current);
                turnstileTokenRef.current = ''; // 使用后立即清空
            }

            const response = await apiClient.post(`/auth/login/access-token`, formData);
            const token = response.data.access_token;

            await auth.login(token);

            toast({
                variant: 'success',
                title: '登录成功',
            });

            navigate('/dashboard');
        } catch (error: any) {
            toast({
                variant: 'error',
                title: '登录失败',
                description: error.response?.data?.detail || '用户名或密码错误',
            });
        } finally {
            setLoading(false);
        }
    };

    // 注册
    const handleRegister = async (e: React.FormEvent) => {
        e.preventDefault();
        
        // Turnstile 流程
        if (systemConfig?.enable_turnstile && !turnstileTokenRef.current) {
            setCurrentAction('register');
            turnstileRef.current?.execute();
            return; // 等待 onVerify 回调
        }
        
        setLoading(true);

        try {
            // 先验证验证码（如果有输入）
            if (verificationCode) {
                const isValid = await verifyCode();
                if (!isValid) {
                    setLoading(false);
                    return;
                }
            }

            // 注册用户
            const registerData: any = {
                username,
                email,
                password
            };

            if (systemConfig?.enable_turnstile) {
                registerData.turnstile_token = turnstileTokenRef.current;
                turnstileTokenRef.current = ''; // 使用后立即清空
            }

            await apiClient.post(`/users/open`, registerData);

            toast({
                variant: 'success',
                title: '注册成功',
                description: '请使用用户名和密码登录',
            });

            // 切换到登录模式
            setIsLogin(true);
            setIsResetPassword(false);
            setVerificationCode('');
        } catch (error: any) {
            toast({
                variant: 'error',
                title: '注册失败',
                description: error.response?.data?.detail || '注册失败，请重试',
            });
        } finally {
            setLoading(false);
        }
    };

    // 重置密码
    const handleResetPassword = async (e: React.FormEvent) => {
        e.preventDefault();

        // Turnstile 流程
        if (systemConfig?.enable_turnstile && !turnstileTokenRef.current) {
            setCurrentAction('reset');
            turnstileRef.current?.execute();
            return; // 等待 onVerify 回调
        }
        
        setLoading(true);

        try {
            // 验证两次密码是否一致
            if (password !== confirmPassword) {
                toast({
                    variant: 'error',
                    title: '密码不一致',
                    description: '两次输入的密码不一致，请重新输入',
                });
                setLoading(false);
                return;
            }

            // 调用重置密码API
            const resetData: any = {
                email: email,
                code: verificationCode,
                new_password: password
            };

            if (systemConfig?.enable_turnstile) {
                resetData.turnstile_token = turnstileTokenRef.current;
                turnstileTokenRef.current = ''; // 使用后立即清空
            }

            await apiClient.post(`/auth/reset-password`, resetData);

            toast({
                variant: 'success',
                title: '密码重置成功',
                description: '请使用新密码登录',
            });

            // 切换到登录模式并清空表单
            setIsLogin(true);
            setIsResetPassword(false);
            setVerificationCode('');
            setPassword('');
            setConfirmPassword('');
        } catch (error: any) {
            toast({
                variant: 'error',
                title: '重置失败',
                description: error.response?.data?.detail || '密码重置失败，请重试',
            });
        } finally {
            setLoading(false);
        }
    };

    // 可重用的人机验证组件
    const TurnstileComponent = () => {
        if (!systemConfig?.enable_turnstile || !systemConfig.turnstile_site_key) {
            return null;
        }
        return (
            <div className="flex justify-center my-4">
                <TurnstileWidget
                    ref={turnstileRef}
                    siteKey={systemConfig.turnstile_site_key}
                    onVerify={handleTurnstileVerify}
                    onError={() => toast({ variant: 'error', title: '人机验证失败，请刷新重试' })}
                    appearance="always" // 保持可见
                />
            </div>
        );
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-primary/10 via-background to-secondary/10 p-4">
            <CaptchaDialog
                open={isCaptchaDialogOpen}
                onOpenChange={setIsCaptchaDialogOpen}
                onSuccess={onCaptchaSuccess}
            />
            {/* Turnstile 组件现在将在下面的表单中被调用 */}
            <div className="bg-card border rounded-xl shadow-2xl w-full max-w-md overflow-hidden">
                {/* 标题区域 */}
                <div className="bg-gradient-to-r from-primary to-primary/80 text-primary-foreground p-8 text-center">
                    <h1 className="text-3xl font-bold mb-2">{systemConfig?.site_name || 'Any API'}</h1>
                    <p className="text-sm opacity-90">
                        {isLogin ? '欢迎回来' : isResetPassword ? '重置您的密码' : '创建新账户'}
                    </p>
                </div>

                {/* 表单区域 */}
                <div className="p-8">
                    {/* 切换标签 */}
                    <div className="flex rounded-lg border p-1 mb-6">
                        <button
                            type="button"
                            onClick={() => {
                                if (systemConfig?.allow_password_login) {
                                    setIsLogin(true);
                                    setIsResetPassword(false);
                                } else {
                                    toast({
                                        variant: 'error',
                                        title: '登录已禁用',
                                        description: '管理员已关闭密码登录功能。',
                                    });
                                }
                            }}
                            className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${isLogin && !isResetPassword
                                ? 'bg-primary text-primary-foreground'
                                : 'text-muted-foreground hover:text-foreground'
                                }`}
                        >
                            登录
                        </button>
                        {systemConfig?.allow_registration && (
                            <button
                                type="button"
                                onClick={() => {
                                    setIsLogin(false);
                                    setIsResetPassword(false);
                                }}
                                className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${!isLogin && !isResetPassword
                                    ? 'bg-primary text-primary-foreground'
                                    : 'text-muted-foreground hover:text-foreground'
                                    }`}
                            >
                                注册
                            </button>
                        )}
                        {systemConfig?.require_email_verification && (
                            <button
                                type="button"
                                onClick={() => {
                                    setIsLogin(false);
                                    setIsResetPassword(true);
                                }}
                                className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${isResetPassword
                                    ? 'bg-primary text-primary-foreground'
                                    : 'text-muted-foreground hover:text-foreground'
                                    }`}
                            >
                                重置密码
                            </button>
                        )}
                    </div>

                    {/* 登录表单 */}
                    {isLogin && !isResetPassword ? (
                        <form onSubmit={handleLogin} className="space-y-4">
                            <div className="space-y-2">
                                <Label htmlFor="login-username">用户名或邮箱</Label>
                                <div className="relative">
                                    <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                                    <Input
                                        id="login-username"
                                        type="text"
                                        placeholder="输入用户名或邮箱"
                                        value={username}
                                        onChange={(e) => setUsername(e.target.value)}
                                        className="pl-10"
                                        required
                                    />
                                </div>
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="login-password">密码</Label>
                                <div className="relative">
                                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                                    <Input
                                        id="login-password"
                                        type="password"
                                        placeholder="输入密码"
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                        className="pl-10"
                                        required
                                    />
                                </div>
                            </div>

                            {/* Turnstile 人机验证 */}
                            <TurnstileComponent />

                            <Button type="submit" className="w-full" disabled={loading}>
                                {loading ? '登录中...' : '登录'}
                            </Button>
                        </form>
                    ) : isResetPassword ? (
                        /* 重置密码表单 */
                        <form onSubmit={handleResetPassword} className="space-y-4">
                            <div className="space-y-2">
                                <Label htmlFor="reset-email">邮箱</Label>
                                <div className="relative">
                                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                                    <Input
                                        ref={emailInputRef}
                                        id="reset-email"
                                        type="email"
                                        placeholder="输入注册时使用的邮箱"
                                        value={email}
                                        onChange={(e) => {
                                            setEmail(e.target.value);
                                            const errorMessage = validateEmail(e.target.value);
                                            (e.target as HTMLInputElement).setCustomValidity(errorMessage);
                                        }}
                                        className="pl-10"
                                        required
                                        onInvalid={(e) => {
                                            const target = e.target as HTMLInputElement;
                                            const errorMessage = validateEmail(target.value);
                                            target.setCustomValidity(errorMessage);
                                        }}
                                        onInput={(e) => (e.target as HTMLInputElement).setCustomValidity('')}
                                    />
                                </div>
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="reset-code">验证码</Label>
                                <div className="flex gap-2">
                                    <div className="relative flex-1">
                                        <Key className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                                        <Input
                                            id="reset-code"
                                            type="text"
                                            placeholder="输入6位验证码"
                                            value={verificationCode}
                                            onChange={(e) => setVerificationCode(e.target.value)}
                                            className="pl-10"
                                            maxLength={6}
                                            required
                                        />
                                    </div>
                                    <Button
                                        type="button"
                                        variant="outline"
                                        onClick={handleSendCode}
                                        disabled={countdown > 0 || !email}
                                        className="shrink-0"
                                    >
                                        {countdown > 0 ? `${countdown}s` : '发送'}
                                    </Button>
                                </div>
                            </div>

                            {/* 图形验证码输入框已被移除，由弹窗代替 */}

                            <div className="space-y-2">
                                <Label htmlFor="reset-new-password">新密码</Label>
                                <div className="relative">
                                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                                    <Input
                                        id="reset-new-password"
                                        type="password"
                                        placeholder="输入新密码（至少6位）"
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                        className="pl-10"
                                        required
                                        minLength={6}
                                        pattern="^(?!\d+$).{6,}$"
                                        onInvalid={(e) => {
                                            const target = e.target as HTMLInputElement;
                                            if (target.value.length < 6) {
                                                target.setCustomValidity('密码长度不能少于6位');
                                            } else {
                                                target.setCustomValidity('密码不能为纯数字');
                                            }
                                        }}
                                        onInput={(e) => (e.target as HTMLInputElement).setCustomValidity('')}
                                    />
                                </div>
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="reset-confirm-password">确认密码</Label>
                                <div className="relative">
                                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                                    <Input
                                        id="reset-confirm-password"
                                        type="password"
                                        placeholder="再次输入新密码"
                                        value={confirmPassword}
                                        onChange={(e) => setConfirmPassword(e.target.value)}
                                        className="pl-10"
                                        required
                                        minLength={6}
                                    />
                                </div>
                            </div>

                            
                            {/* Turnstile 人机验证 */}
                            <TurnstileComponent />
                            
                            <Button type="submit" className="w-full" disabled={loading}>
                                 {loading ? '重置中...' : '重置密码'}
                            </Button>
                         </form>
                    ) : (
                        /* 注册表单 */
                        configLoading ? (
                            <div className="text-center text-muted-foreground">加载配置中...</div>
                        ) : (
                            <form onSubmit={handleRegister} className="space-y-4">
                                <div className="space-y-2">
                                    <Label htmlFor="register-username">用户名</Label>
                                    <div className="relative">
                                        <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                                        <Input
                                            id="register-username"
                                            type="text"
                                            placeholder="输入用户名"
                                            value={username}
                                            onChange={(e) => setUsername(e.target.value)}
                                            className="pl-10"
                                            required
                                            minLength={4}
                                            pattern="^[a-zA-Z0-9]+$"
                                            onInvalid={(e) => {
                                                const target = e.target as HTMLInputElement;
                                                if (target.value.length < 4) {
                                                    target.setCustomValidity('用户名长度不能少于4位');
                                                } else {
                                                    target.setCustomValidity('用户名只能包含字母和数字');
                                                }
                                            }}
                                            onInput={(e) => (e.target as HTMLInputElement).setCustomValidity('')}
                                        />
                                    </div>
                                </div>

                                <div className="space-y-2">
                                    <Label htmlFor="register-email">邮箱</Label>
                                    <div className="relative">
                                        <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                                        <Input
                                            ref={emailInputRef}
                                            id="register-email"
                                            type="email"
                                            placeholder="输入邮箱地址"
                                            value={email}
                                            onChange={(e) => {
                                                setEmail(e.target.value);
                                                const errorMessage = validateEmail(e.target.value);
                                                (e.target as HTMLInputElement).setCustomValidity(errorMessage);
                                            }}
                                            className="pl-10"
                                            required
                                            onInvalid={(e) => {
                                                const target = e.target as HTMLInputElement;
                                                const errorMessage = validateEmail(target.value);
                                                target.setCustomValidity(errorMessage);
                                            }}
                                            onInput={(e) => (e.target as HTMLInputElement).setCustomValidity('')}
                                        />
                                    </div>
                                </div>

                                {systemConfig?.require_email_verification && (
                                    <div className="space-y-2">
                                        <Label htmlFor="register-code">验证码</Label>
                                        <div className="flex gap-2">
                                            <div className="relative flex-1">
                                                <Key className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                                                <Input
                                                    id="register-code"
                                                    type="text"
                                                    placeholder="输入6位验证码"
                                                    value={verificationCode}
                                                    onChange={(e) => setVerificationCode(e.target.value)}
                                                    className="pl-10"
                                                    maxLength={6}
                                                />
                                            </div>
                                            <Button
                                                type="button"
                                                variant="outline"
                                                onClick={handleSendCode}
                                                disabled={countdown > 0 || !email}
                                                className="shrink-0"
                                            >
                                                {countdown > 0 ? `${countdown}s` : '发送'}
                                            </Button>
                                        </div>
                                    </div>
                                )}
                                
                                {/* 图形验证码输入框已被移除，由弹窗代替 */}

                                <div className="space-y-2">
                                    <Label htmlFor="register-password">密码</Label>
                                    <div className="relative">
                                        <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                                        <Input
                                            id="register-password"
                                            type="password"
                                            placeholder="输入密码（至少6位）"
                                            value={password}
                                            onChange={(e) => setPassword(e.target.value)}
                                            className="pl-10"
                                            required
                                            minLength={6}
                                            pattern="^(?!\d+$).{6,}$"
                                            onInvalid={(e) => {
                                                const target = e.target as HTMLInputElement;
                                                if (target.value.length < 6) {
                                                    target.setCustomValidity('密码长度不能少于6位');
                                                } else {
                                                    target.setCustomValidity('密码不能为纯数字');
                                                }
                                            }}
                                            onInput={(e) => (e.target as HTMLInputElement).setCustomValidity('')}
                                        />
                                    </div>
                                </div>


                                {/* Turnstile 人机验证 */}
                                <TurnstileComponent />

                                <Button type="submit" className="w-full" disabled={loading}>
                                    {loading ? '注册中...' : '注册'}
                                </Button>
                            </form>
                        )
                    )}
                </div>
            </div>
        </div>
    );
}
