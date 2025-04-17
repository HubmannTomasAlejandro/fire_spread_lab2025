	.file	"spread_functions.cpp"
	.text
#APP
	.globl _ZSt21ios_base_library_initv
#NO_APP
	.section	.text._ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv,"axG",@progbits,_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
	.type	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv, @function
_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv:
.LFB11338:
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	movq	%rdi, %rbx
	movq	(%rdi), %rdi
	testq	%rdi, %rdi
	je	.L6
	movq	32(%rbx), %rsi
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
	movq	$0, (%rbx)
	movl	$0, 8(%rbx)
	movq	$0, 16(%rbx)
	movl	$0, 24(%rbx)
	movq	$0, 32(%rbx)
.L6:
	popq	%rbx
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE11338:
	.size	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv, .-_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC1:
	.string	"vector::_M_realloc_append"
	.section	.text.unlikely,"ax",@progbits
.LCOLDB13:
	.text
.LHOTB13:
	.p2align 4
	.globl	_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff
	.type	_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff, @function
_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff:
.LFB10559:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA10559
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	pushq	-8(%r10)
	pushq	%rbp
	movq	%rsp, %rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	pushq	%rbx
	movq	%rdx, %r14
	subq	$544, %rsp
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	movq	8(%rsi), %rax
	movq	%r10, -512(%rbp)
	movq	%rdi, -544(%rbp)
	movq	%rsi, -288(%rbp)
	vmovss	%xmm0, -468(%rbp)
	vmovss	%xmm1, -480(%rbp)
	vmovss	%xmm2, -268(%rbp)
	vmovss	%xmm3, -484(%rbp)
	movq	%rax, -352(%rbp)
	movq	(%rsi), %rax
	movq	%rax, -304(%rbp)
	movq	8(%rdx), %rax
	movq	%rax, -256(%rbp)
	movq	(%rdx), %rdx
	movq	$0, -496(%rbp)
	movq	$0, -384(%rbp)
	movq	$0, -360(%rbp)
	subq	%rdx, %rax
	movq	%rdx, -264(%rbp)
	movq	%rax, %r13
	sarq	$4, %r13
	movq	%r13, -368(%rbp)
	testq	%rax, %rax
	je	.L10
	xorl	%r12d, %r12d
	jmp	.L20
	.p2align 6
	.p2align 4
	.p2align 3
.L183:
	vmovdqu	(%r15), %xmm0
	addq	$16, %rax
	incq	%r12
	vmovdqu	%xmm0, -16(%rax)
	movq	%rax, -384(%rbp)
	cmpq	%r13, %r12
	jnb	.L10
.L184:
	movq	(%r14), %rdx
.L20:
	movq	%r12, %rax
	movq	-496(%rbp), %rbx
	salq	$4, %rax
	leaq	(%rdx,%rax), %r15
	movq	-384(%rbp), %rax
	cmpq	%rbx, %rax
	jne	.L183
	movq	-360(%rbp), %rax
	movq	%rbx, %rcx
	movabsq	$576460752303423487, %rbx
	subq	%rax, %rcx
	movq	%rcx, %rax
	sarq	$4, %rax
	cmpq	%rbx, %rax
	je	.L173
	testq	%rax, %rax
	movl	$1, %edx
	cmovne	%rax, %rdx
	addq	%rdx, %rax
	jc	.L15
	movabsq	$576460752303423487, %rdx
	cmpq	%rdx, %rax
	cmova	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rbx
.L16:
	movq	%rbx, %rdi
	movq	%rcx, -248(%rbp)
.LEHB0:
	call	_Znwm@PLT
.LEHE0:
	vmovdqu	(%r15), %xmm0
	movq	-248(%rbp), %rcx
	movq	-384(%rbp), %rsi
	movq	-360(%rbp), %rdx
	vmovdqu	%xmm0, (%rax,%rcx)
	cmpq	%rdx, %rsi
	je	.L111
	movq	%rax, %rcx
	.p2align 5
	.p2align 4
	.p2align 3
.L18:
	vmovdqu	(%rdx), %xmm0
	addq	$16, %rdx
	addq	$16, %rcx
	vmovdqu	%xmm0, -16(%rcx)
	cmpq	%rdx, %rsi
	jne	.L18
	movq	-384(%rbp), %rdx
	movq	-360(%rbp), %rdi
	subq	%rdi, %rdx
	addq	%rax, %rdx
.L17:
	addq	$16, %rdx
	movq	%rdx, -384(%rbp)
	movq	-360(%rbp), %rdx
	testq	%rdx, %rdx
	je	.L19
	movq	-496(%rbp), %rsi
	movq	%rdx, %rdi
	movq	%rax, -248(%rbp)
	subq	%rdx, %rsi
	call	_ZdlPvm@PLT
	movq	-248(%rbp), %rax
.L19:
	addq	%rax, %rbx
	incq	%r12
	movq	%rax, -360(%rbp)
	movq	%rbx, -496(%rbp)
	cmpq	%r13, %r12
	jb	.L184
.L10:
	movl	$8, %edi
.LEHB1:
	call	_Znwm@PLT
.LEHE1:
	vmovq	-304(%rbp), %xmm6
	movq	%r13, (%rax)
	movq	%rax, -536(%rbp)
	addq	$8, %rax
	movq	%rax, -504(%rbp)
	movl	$0, -152(%rbp)
	movq	-352(%rbp), %rax
	movq	-304(%rbp), %rbx
	imulq	%rax, %rbx
	vpinsrq	$1, %rax, %xmm6, %xmm6
	vmovdqa	%xmm6, -576(%rbp)
	vmovdqa	%xmm6, -176(%rbp)
	testq	%rbx, %rbx
	jne	.L185
	movl	$0, -520(%rbp)
	movq	$0, -552(%rbp)
	movq	$0, -280(%rbp)
	movq	$0, -560(%rbp)
.L21:
	movq	-264(%rbp), %rbx
	cmpq	%rbx, -256(%rbp)
	je	.L31
	movq	(%r14), %rcx
	xorl	%esi, %esi
	movl	$1, %r8d
	.p2align 4
	.p2align 3
.L32:
	movq	-304(%rbp), %rax
	imulq	8(%rcx), %rax
	movq	-280(%rbp), %rbx
	addq	(%rcx), %rax
	leaq	63(%rax), %rdx
	cmovns	%rax, %rdx
	sarq	$6, %rdx
	leaq	(%rbx,%rdx,8), %rdi
	cqto
	shrq	$58, %rdx
	addq	%rdx, %rax
	andl	$63, %eax
	subq	%rdx, %rax
	js	.L186
	incq	%rsi
	shlx	%rax, %r8, %rax
	addq	$16, %rcx
	orq	%rax, (%rdi)
	cmpq	%r13, %rsi
	jb	.L32
.L31:
	cmpq	$-1, %r13
	je	.L187
.L25:
	vmovss	.LC2(%rip), %xmm6
	movq	-504(%rbp), %rax
	movq	$0, -248(%rbp)
	vdivss	-468(%rbp), %xmm6, %xmm4
	movq	%rax, -528(%rbp)
	movq	-368(%rbp), %rax
	vdivss	-268(%rbp), %xmm6, %xmm6
	vmovss	%xmm4, -476(%rbp)
	vmovss	%xmm6, -472(%rbp)
	.p2align 4
	.p2align 3
.L33:
	leaq	-240(%rbp), %rbx
	movq	%rax, -464(%rbp)
	movq	%rbx, -376(%rbp)
	cmpq	%rax, -248(%rbp)
	jnb	.L188
	.p2align 4
	.p2align 3
.L60:
	movq	-248(%rbp), %rax
	movq	-360(%rbp), %rbx
	movq	-288(%rbp), %rsi
	movq	-376(%rbp), %rdi
	salq	$4, %rax
	addq	%rbx, %rax
	movq	(%rax), %r12
	movq	8(%rax), %rbx
	movq	%r12, %rdx
	movq	%rbx, %rcx
	vzeroupper
.LEHB2:
	call	_ZNK9LandscapeixESt4pairImmE@PLT
	movl	%r12d, %r9d
	leaq	_ZL5MOVES(%rip), %r13
	leaq	-112(%rbp), %r12
	movq	-304(%rbp), %r10
	movl	%ebx, %r8d
	movq	%r12, %rdx
	xorl	%ebx, %ebx
	xorl	%eax, %eax
	leaq	4(%r13), %r11
	movl	$1, %r14d
	movq	%r12, -256(%rbp)
	jmp	.L39
	.p2align 4
	.p2align 3
.L190:
	movl	%ecx, %esi
	orl	%edi, %esi
	js	.L34
	movl	-352(%rbp), %esi
	cmpl	%esi, %ecx
	jge	.L34
	movslq	%ecx, %rcx
	movq	%r10, %rsi
	movq	-280(%rbp), %r15
	movl	%edi, (%rdx)
	imulq	%rcx, %rsi
	movslq	%edi, %rdi
	leaq	63(%rdi,%rsi), %r12
	addq	%rdi, %rsi
	cmovns	%rsi, %r12
	sarq	$6, %r12
	leaq	(%r15,%r12,8), %r15
	movq	%rsi, %r12
	sarq	$63, %r12
	shrq	$58, %r12
	addq	%r12, %rsi
	andl	$63, %esi
	subq	%r12, %rsi
	jns	.L35
	addq	$64, %rsi
	subq	$8, %r15
.L35:
	shlx	%rsi, %r14, %rsi
	andq	(%r15), %rsi
	jne	.L36
	movq	-288(%rbp), %r15
	imulq	96(%r15), %rcx
	leaq	63(%rdi,%rcx), %rsi
	addq	%rdi, %rcx
	movq	112(%r15), %rdi
	cmovns	%rcx, %rsi
	sarq	$6, %rsi
	leaq	(%rdi,%rsi,8), %rdi
	movq	%rcx, %rsi
	sarq	$63, %rsi
	shrq	$58, %rsi
	addq	%rsi, %rcx
	andl	$63, %ecx
	subq	%rsi, %rcx
	jns	.L37
	addq	$64, %rcx
	subq	$8, %rdi
.L37:
	movl	$1, %esi
	shlx	%rax, %rsi, %r12
	shlx	%rcx, %rsi, %rsi
	andq	(%rdi), %rsi
	je	.L36
	incq	%rax
	orq	%r12, %rbx
	addq	$4, %rdx
	cmpq	$8, %rax
	je	.L189
.L39:
	movl	0(%r13,%rax,8), %edi
	movl	(%r11,%rax,8), %ecx
	addl	%r9d, %edi
	addl	%r8d, %ecx
	movl	%ecx, 32(%rdx)
	cmpl	%r10d, %edi
	jl	.L190
.L34:
	movl	%r9d, (%rdx)
	movl	%r8d, 32(%rdx)
.L36:
	shlx	%rax, %r14, %rcx
	incq	%rax
	addq	$4, %rdx
	andn	%rbx, %rcx, %rbx
	cmpq	$8, %rax
	jne	.L39
.L189:
	movq	-256(%rbp), %r12
	testq	%rbx, %rbx
	jne	.L191
.L59:
	incq	-248(%rbp)
	movq	-368(%rbp), %rbx
	movq	-248(%rbp), %rax
	cmpq	%rbx, %rax
	jne	.L60
	movq	-504(%rbp), %rax
	movq	-528(%rbp), %rbx
	cmpq	%rbx, %rax
	je	.L62
	movq	-464(%rbp), %rbx
	addq	$8, %rax
	movq	%rbx, -8(%rax)
	movq	%rax, -504(%rbp)
.L64:
	movq	-368(%rbp), %rax
	movq	%rax, -248(%rbp)
	cmpq	%rax, -464(%rbp)
	je	.L63
	movq	-464(%rbp), %rax
	movq	%rax, -368(%rbp)
	jmp	.L33
	.p2align 4
	.p2align 3
.L191:
	movq	-512(%rbp), %rax
	movq	-288(%rbp), %rdx
	leaq	-208(%rbp), %r14
	xorl	%r13d, %r13d
	vmovsd	16(%rax), %xmm5
	vmovsd	8(%rax), %xmm7
	vmovsd	64(%rax), %xmm6
	movq	16(%rdx), %rdi
	movq	32(%rdx), %rsi
	movq	168(%rdx), %r8
	movq	208(%rdx), %r9
	movq	232(%rdx), %r10
	movq	248(%rdx), %r11
	movq	%rdi, -408(%rbp)
	movswl	-240(%rbp), %edi
	movq	%rsi, -416(%rbp)
	movq	192(%rdx), %rsi
	movq	%r9, %r15
	vmovsd	%xmm5, -440(%rbp)
	vmovsd	56(%rax), %xmm5
	vmovsd	%xmm7, -432(%rbp)
	vmovsd	24(%rax), %xmm7
	vmovsd	%xmm6, -456(%rbp)
	movl	%edi, -392(%rbp)
	movq	%rsi, -296(%rbp)
	movq	152(%rdx), %rdi
	vmovsd	%xmm5, -448(%rbp)
	vxorps	%xmm5, %xmm5, %xmm5
	vmovsd	%xmm7, -400(%rbp)
	vmovss	-236(%rbp), %xmm7
	vcvtsd2ss	(%rax), %xmm5, %xmm3
	vcvtss2sd	%xmm3, %xmm3, %xmm0
	vaddsd	48(%rax), %xmm0, %xmm1
	vmovss	%xmm7, -420(%rbp)
	vcvtsd2ss	%xmm1, %xmm1, %xmm6
	vaddsd	40(%rax), %xmm0, %xmm1
	vmovss	%xmm6, -488(%rbp)
	vcvtsd2ss	%xmm1, %xmm1, %xmm4
	vmovss	%xmm4, -424(%rbp)
	vaddsd	32(%rax), %xmm0, %xmm0
	movq	%r14, %rax
	vmovss	%xmm3, -516(%rbp)
	movq	%r8, %r14
	movq	%rbx, %r8
	movq	%rdi, %rbx
	movq	%rax, %rdi
	vcvtsd2ss	%xmm0, %xmm0, %xmm7
	vmovss	%xmm7, -388(%rbp)
	jmp	.L46
	.p2align 4
	.p2align 3
.L194:
	vmovss	-424(%rbp), %xmm5
	cmpb	$2, %cl
	je	.L42
	vmovss	-488(%rbp), %xmm5
	cmpb	$3, %cl
	je	.L42
	cmpb	$4, %cl
	jne	.L192
	vxorps	%xmm0, %xmm0, %xmm0
.L45:
	vmovss	%xmm0, (%rdi,%r13,4)
	incq	%r13
	cmpq	$8, %r13
	je	.L193
.L46:
	movslq	32(%r12,%r13,4), %rax
	movslq	(%r12,%r13,4), %rdx
	movq	%rbx, %rcx
	vmovss	-388(%rbp), %xmm5
	imulq	%rax, %rcx
	leaq	(%r14,%rdx), %rsi
	movzbl	(%rsi,%rcx), %ecx
	cmpb	$1, %cl
	jne	.L194
.L42:
	movq	-296(%rbp), %rcx
	imulq	%rax, %rcx
	addq	%rdx, %rcx
	vmovss	(%r15,%rcx,4), %xmm3
	movq	%r10, %rcx
	imulq	%rax, %rcx
	addq	%rdx, %rcx
	vmovss	(%r11,%rcx,4), %xmm2


.L108:
	btq	%r13, %r8
	vmovss	%xmm5, -268(%rbp)
	vmovss	%xmm2, -264(%rbp)
	vmovss	%xmm3, -256(%rbp)
	vxorps	%xmm0, %xmm0, %xmm0
	jnc	.L45
	movq	%rdi, -344(%rbp)
	movq	-408(%rbp), %rdi
	vxorps	%xmm6, %xmm6, %xmm6
	movq	%r8, -336(%rbp)
	movq	%r11, -328(%rbp)
	movq	%r10, -320(%rbp)
	imulq	%rdi, %rax
	addq	%rdx, %rax
	movq	-416(%rbp), %rdx
	movswl	(%rdx,%rax,2), %eax
	movl	%eax, %edx
	movl	%eax, -312(%rbp)
	movl	-392(%rbp), %eax
	subl	%eax, %edx
	vcvtsi2ssl	%edx, %xmm6, %xmm1
	leaq	_ZL6ANGLES(%rip), %rdx
	vmulss	-476(%rbp), %xmm1, %xmm4
	vmovss	%xmm1, -308(%rbp)
	vmovss	(%rdx,%r13,4), %xmm0
	vsubss	-420(%rbp), %xmm0, %xmm0
	vmovss	%xmm4, -272(%rbp)
	call	cosf@PLT
	vmovss	-256(%rbp), %xmm3
	vxorps	%xmm6, %xmm6, %xmm6
	vmovss	-268(%rbp), %xmm5
	vmovss	-264(%rbp), %xmm2
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	movl	-312(%rbp), %eax
	vmovss	-272(%rbp), %xmm4
	vmovss	-308(%rbp), %xmm1
	vmovss	.LC6(%rip), %xmm7
	vcvtss2sd	%xmm3, %xmm3, %xmm3
	vcvtss2sd	%xmm5, %xmm5, %xmm5
	vfmadd132sd	-448(%rbp), %xmm5, %xmm3
	vandps	.LC3(%rip), %xmm4, %xmm5
	vcvtss2sd	%xmm2, %xmm2, %xmm2
	vorps	.LC4(%rip), %xmm5, %xmm5
	vcvtsd2ss	%xmm3, %xmm3, %xmm3
	vcvtss2sd	%xmm3, %xmm3, %xmm3
	vfmadd132sd	-456(%rbp), %xmm3, %xmm2
	vcvtsi2ssl	%eax, %xmm6, %xmm3
	vsubss	-480(%rbp), %xmm3, %xmm3
	vmulss	-472(%rbp), %xmm3, %xmm3
	vcvtsd2ss	%xmm2, %xmm2, %xmm2
	vcvtss2sd	%xmm3, %xmm3, %xmm3
	vcvtss2sd	%xmm2, %xmm2, %xmm2
	vfmadd231sd	-440(%rbp), %xmm3, %xmm2
	vmovss	.LC2(%rip), %xmm3
	vfmadd231ss	%xmm4, %xmm4, %xmm3
	vandps	.LC5(%rip), %xmm4, %xmm4
	vsqrtss	%xmm3, %xmm3, %xmm3
	vmulss	-468(%rbp), %xmm3, %xmm3
	vcmpless	%xmm4, %xmm7, %xmm4
	vdivss	%xmm3, %xmm1, %xmm1
	vblendvps	%xmm4, %xmm5, %xmm1, %xmm1
	vcvtss2sd	%xmm1, %xmm1, %xmm1
	vmulsd	-400(%rbp), %xmm1, %xmm1
	vfmadd132sd	-432(%rbp), %xmm1, %xmm0
	vaddsd	%xmm2, %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	vxorps	.LC3(%rip), %xmm0, %xmm0
	call	expf@PLT
	vaddss	.LC2(%rip), %xmm0, %xmm0
	vmovss	-484(%rbp), %xmm4
	movq	-320(%rbp), %r10
	movq	-328(%rbp), %r11
	movq	-344(%rbp), %rdi
	movq	-336(%rbp), %r8
	vdivss	%xmm0, %xmm4, %xmm0
	vmovss	%xmm0, (%rdi,%r13,4)
	incq	%r13
	cmpq	$8, %r13
	jne	.L46


.L193:
	vmovdqa	rng(%rip), %ymm1
	vbroadcastss	.LC9(%rip), %ymm2
	movl	$65535, %eax
	vpslld	$13, %ymm1, %ymm0
	vpxor	%ymm1, %ymm0, %ymm0
	vpsrld	$17, %ymm0, %ymm1
	vpxor	%ymm0, %ymm1, %ymm1
	vpslld	$5, %ymm1, %ymm0
	vpxor	%ymm1, %ymm0, %ymm0
	vmovd	%eax, %xmm1
	vpbroadcastd	%xmm1, %ymm1
	vmovdqa	%ymm0, rng(%rip)
	vpand	%ymm1, %ymm0, %ymm1
	vpsrld	$16, %ymm0, %ymm0
	vcvtdq2ps	%ymm0, %ymm0
	vcvtdq2ps	%ymm1, %ymm1
	vfmadd132ps	%ymm2, %ymm1, %ymm0
	vbroadcastss	.LC11(%rip), %ymm1
	vmulps	%ymm1, %ymm0, %ymm0
	vcmpps	$17, -208(%rbp), %ymm0, %ymm0
	vmovmskps	%ymm0, %r14d
	testb	%r14b, %r14b
	je	.L59
	movzbl	%r14b, %r14d
	xorl	%r13d, %r13d
	movl	$1, %edx
	jmp	.L58
	.p2align 4
	.p2align 4
	.p2align 3
.L47:
	incq	%r13
	cmpq	$8, %r13
	je	.L59
.L58:
	btl	%r13d, %r14d
	jnc	.L47
	movq	-384(%rbp), %rax
	movq	-496(%rbp), %rsi
	movslq	(%r12,%r13,4), %r15
	movslq	32(%r12,%r13,4), %rbx
	cmpq	%rsi, %rax
	je	.L48
	vmovq	%r15, %xmm5
	addq	$16, %rax
	vpinsrq	$1, %rbx, %xmm5, %xmm0
	vmovdqu	%xmm0, -16(%rax)
	movq	%rax, -384(%rbp)
.L49:
	movq	-304(%rbp), %rax
	movq	-280(%rbp), %rsi
	imulq	%rax, %rbx
	leaq	63(%r15,%rbx), %rax
	addq	%r15, %rbx
	cmovns	%rbx, %rax
	sarq	$6, %rax
	leaq	(%rsi,%rax,8), %rcx
	movq	%rbx, %rax
	sarq	$63, %rax
	shrq	$58, %rax
	addq	%rax, %rbx
	andl	$63, %ebx
	subq	%rax, %rbx
	jns	.L57
	addq	$64, %rbx
	subq	$8, %rcx
.L57:
	shlx	%rbx, %rdx, %rbx
	incq	-464(%rbp)
	orq	%rbx, (%rcx)
	jmp	.L47
.L48:
	movq	-360(%rbp), %rax
	movq	%rsi, %rcx
	movabsq	$576460752303423487, %rsi
	subq	%rax, %rcx
	movq	%rcx, %rax
	sarq	$4, %rax
	cmpq	%rsi, %rax
	je	.L174
	testq	%rax, %rax
	movl	$1, %esi
	cmovne	%rax, %rsi
	addq	%rsi, %rax
	jc	.L52
	movabsq	$576460752303423487, %rsi
	cmpq	%rsi, %rax
	cmova	%rsi, %rax
	salq	$4, %rax
	movq	%rax, -256(%rbp)
	movq	%rax, %rdi
.L53:
	movq	%rcx, -264(%rbp)
	vzeroupper
	call	_Znwm@PLT
	movq	-264(%rbp), %rcx
	vmovq	%r15, %xmm6
	movq	-384(%rbp), %rdi
	movl	$1, %edx
	vpinsrq	$1, %rbx, %xmm6, %xmm0
	vmovdqu	%xmm0, (%rax,%rcx)
	movq	-360(%rbp), %rcx
	cmpq	%rcx, %rdi
	je	.L116
	movq	%rax, %rsi
	.p2align 5
	.p2align 4
	.p2align 3
.L55:
	vmovdqu	(%rcx), %xmm0
	addq	$16, %rcx
	addq	$16, %rsi
	vmovdqu	%xmm0, -16(%rsi)
	cmpq	%rcx, %rdi
	jne	.L55
	movq	-384(%rbp), %rcx
	movq	-360(%rbp), %rdi
	subq	%rdi, %rcx
	addq	%rax, %rcx
.L54:
	leaq	16(%rcx), %rsi
	movq	%rsi, -384(%rbp)
	testq	%rdi, %rdi
	je	.L56
	movq	-496(%rbp), %rsi
	movq	%rax, -264(%rbp)
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
	movq	-264(%rbp), %rax
	movl	$1, %edx
.L56:
	movq	-256(%rbp), %rsi
	movq	%rax, -360(%rbp)
	addq	%rax, %rsi
	movq	%rsi, -496(%rbp)
	jmp	.L49
.L186:
	addq	$64, %rax
	incq	%rsi
	addq	$16, %rcx
	shlx	%rax, %r8, %rax
	orq	%rax, -8(%rdi)
	cmpq	%r13, %rsi
	jb	.L32
	cmpq	$-1, %r13
	jne	.L25
.L187:
	movq	-504(%rbp), %rax
	movl	$8, %r12d
	movq	%rax, -528(%rbp)
	jmp	.L26
	.p2align 4
	.p2align 3
.L62:
	movq	-536(%rbp), %rax
	movq	-504(%rbp), %r13
	movabsq	$1152921504606846975, %rdx
	subq	%rax, %r13
	movq	%r13, %rax
	sarq	$3, %rax
	cmpq	%rdx, %rax
	je	.L175
	testq	%rax, %rax
	movl	$1, %edx
	cmovne	%rax, %rdx
	addq	%rdx, %rax
	jc	.L67
	movabsq	$1152921504606846975, %rdx
	cmpq	%rdx, %rax
	cmova	%rdx, %rax
	leaq	0(,%rax,8), %rbx
.L68:
	movq	%rbx, %rdi
	vzeroupper
	call	_Znwm@PLT
.LEHE2:
	movq	%rax, %r12
	movq	-464(%rbp), %rax
	leaq	8(%r13), %r14
	movq	%rax, (%r12,%r13)
	testq	%r13, %r13
	jg	.L195
	cmpq	$0, -536(%rbp)
	leaq	(%r12,%r14), %rax
	movq	%rax, -504(%rbp)
	jne	.L70
.L71:
	leaq	(%r12,%rbx), %rax
	movq	%r12, -536(%rbp)
	movq	%rax, -528(%rbp)
	jmp	.L64
.L188:
	movq	-504(%rbp), %rax
	movq	-528(%rbp), %rbx
	cmpq	%rbx, %rax
	je	.L62
	movq	-368(%rbp), %rbx
	addq	$8, %rax
	movq	%rbx, -8(%rax)
	movq	%rax, -504(%rbp)
.L63:
	movq	-528(%rbp), %r12
	movq	-536(%rbp), %rax
	subq	%rax, %r12
.L26:
	vbroadcasti128	-576(%rbp), %ymm0
	movq	-544(%rbp), %rax
	movq	-560(%rbp), %rbx
	movl	-520(%rbp), %r13d
	movq	$0, 32(%rax)
	movl	$0, 40(%rax)
	movq	$0, 48(%rax)
	movl	$0, 56(%rax)
	movq	$0, 64(%rax)
	vmovdqu	%ymm0, (%rax)
	movq	-280(%rbp), %rax
	subq	%rax, %rbx
	leaq	0(%r13,%rbx,8), %r14
	testq	%r14, %r14
	jne	.L73
	xorl	%ecx, %ecx
	vzeroupper
.L74:
	movl	-520(%rbp), %eax
	movq	-560(%rbp), %rdi
	addq	%rbx, %rcx
	xorl	%edx, %edx
	movl	$1, %r8d
	testl	%eax, %eax
	jne	.L78
	jmp	.L85
	.p2align 6
	.p2align 4
	.p2align 3
.L196:
	incl	%edx
	decq	%r13
	je	.L85
.L78:
	movq	(%rcx), %rsi
	movq	(%rdi), %r10
	shlx	%rdx, %r8, %rax
	andq	%rax, %r10
	movq	%rsi, %r9
	orq	%rax, %r9
	andn	%rsi, %rax, %rax
	testq	%r10, %r10
	cmovne	%r9, %rax
	movq	%rax, (%rcx)
	cmpl	$63, %edx
	jne	.L196
	addq	$8, %rdi
	addq	$8, %rcx
	xorl	%edx, %edx
	decq	%r13
	jne	.L78
.L85:
	movq	-544(%rbp), %rax
	vpxor	%xmm0, %xmm0, %xmm0
	movq	-384(%rbp), %rbx
	vmovdqu	%xmm0, 72(%rax)
	movq	$0, 88(%rax)
	movq	-360(%rbp), %rax
	subq	%rax, %rbx
	je	.L197
	movabsq	$9223372036854775792, %rax
	cmpq	%rbx, %rax
	jb	.L198
	movq	%rbx, %rdi
.LEHB3:
	call	_Znwm@PLT
.LEHE3:
	movq	%rax, %rcx
.L80:
	movq	-544(%rbp), %rax
	addq	%rcx, %rbx
	movq	-360(%rbp), %rsi
	movq	%rbx, 88(%rax)
	movq	-384(%rbp), %rbx
	movq	%rcx, 72(%rax)
	cmpq	%rsi, %rbx
	je	.L89
	subq	%rsi, %rbx
	movq	%rcx, %rdi
	movq	%rbx, %rdx
	call	memcpy@PLT
	movq	%rax, %rcx
	addq	%rbx, %rcx
.L89:
	movq	-544(%rbp), %rax
	vpxor	%xmm0, %xmm0, %xmm0
	movq	-504(%rbp), %rbx
	movq	%rcx, 80(%rax)
	vmovdqu	%xmm0, 96(%rax)
	movq	$0, 112(%rax)
	movq	-536(%rbp), %rax
	subq	%rax, %rbx
	je	.L90
	movabsq	$9223372036854775800, %rax
	cmpq	%rbx, %rax
	jb	.L199
	movq	%rbx, %rdi
.LEHB4:
	call	_Znwm@PLT
.LEHE4:
	movq	%rax, %rdi
	vmovq	%rax, %xmm4
	movq	-544(%rbp), %rax
	vpunpcklqdq	%xmm4, %xmm4, %xmm0
	leaq	(%rdi,%rbx), %r13
	vmovdqu	%xmm0, 96(%rax)
	movq	%r13, 112(%rax)
	cmpq	$8, %rbx
	jle	.L200
	movq	%rbx, %rdx
	movq	-536(%rbp), %rsi
	call	memcpy@PLT
	cmpq	$0, -280(%rbp)
	movq	-544(%rbp), %rax
	movq	%r13, 104(%rax)
	je	.L99
.L98:
	movq	-552(%rbp), %rsi
	movq	-280(%rbp), %rdi
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
.L99:
	movq	-536(%rbp), %rdi
	movq	%r12, %rsi
	call	_ZdlPvm@PLT
.L100:
	movq	-360(%rbp), %rdi
	testq	%rdi, %rdi
	je	.L9
	movq	-496(%rbp), %rsi
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
.L9:
	movq	-544(%rbp), %rax
	addq	$544, %rsp
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
.L195:
	.cfi_restore_state
	movq	-536(%rbp), %rsi
	movq	%r13, %rdx
	movq	%r12, %rdi
	call	memcpy@PLT
	leaq	(%r12,%r14), %rax
	movq	%rax, -504(%rbp)
.L70:
	movq	-528(%rbp), %rsi
	movq	-536(%rbp), %rdi
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
	jmp	.L71
.L197:
	xorl	%ecx, %ecx
	jmp	.L80
.L90:
	cmpq	$0, -280(%rbp)
	movq	-544(%rbp), %rax
	movq	$0, 112(%rax)
	je	.L95
	movq	-552(%rbp), %rsi
	movq	-280(%rbp), %rdi
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
.L95:
	cmpq	$0, -536(%rbp)
	je	.L100
	jmp	.L99
	.p2align 4
	.p2align 3
.L116:
	movq	-360(%rbp), %rdi
	movq	%rax, %rcx
	jmp	.L54
.L73:
	leaq	63(%r14), %rdx
	movq	%rdx, %r15
	movq	%rdx, -248(%rbp)
	shrq	$6, %r15
	salq	$3, %r15
	movq	%r15, %rdi
	vzeroupper
.LEHB5:
	call	_Znwm@PLT
.LEHE5:
	movq	%rax, %rcx
	addq	%rax, %r15
	movq	-544(%rbp), %rax
	testq	%r14, %r14
	movq	%r14, %rdx
	movq	%r15, 64(%rax)
	movq	%rcx, 32(%rax)
	movl	$0, 40(%rax)
	movq	-248(%rbp), %rax
	cmovns	%r14, %rax
	sarq	$63, %rdx
	sarq	$6, %rax
	shrq	$58, %rdx
	leaq	(%rcx,%rax,8), %rsi
	leaq	(%r14,%rdx), %rax
	andl	$63, %eax
	subq	%rdx, %rax
	js	.L201
.L75:
	movq	-544(%rbp), %rdx
	movq	%rsi, 48(%rdx)
	movl	%eax, 56(%rdx)
	cmpq	$8, %rbx
	jle	.L76
	movq	%rcx, %rdi
	movq	%rbx, %rdx
	movq	-280(%rbp), %rsi
	call	memcpy@PLT
	movq	%rax, %rcx
	jmp	.L74
.L185:
	leaq	63(%rbx), %r15
	movq	%r15, %r12
	shrq	$6, %r12
	salq	$3, %r12
	movq	%r12, %rdi
.LEHB6:
	call	_Znwm@PLT
.LEHE6:
	movq	%rax, %rdx
	movq	%rax, -280(%rbp)
	leaq	(%rax,%r12), %rax
	testq	%rbx, %rbx
	movq	%rax, -552(%rbp)
	movq	%r15, %rax
	cmovns	%rbx, %rax
	sarq	$6, %rax
	leaq	(%rdx,%rax,8), %rax
	movq	%rbx, %rdx
	sarq	$63, %rdx
	movq	%rax, -560(%rbp)
	shrq	$58, %rdx
	leaq	(%rbx,%rdx), %rax
	andl	$63, %eax
	subq	%rdx, %rax
	js	.L22
.L182:
	movq	-280(%rbp), %rdi
	movq	%r12, %rdx
	xorl	%esi, %esi
	movl	%eax, -520(%rbp)
	movl	$0, -152(%rbp)
	call	memset@PLT
	jmp	.L21
.L22:
	subq	$8, -560(%rbp)
	addl	$64, %eax
	jmp	.L182
.L201:
	addq	$64, %rax
	subq	$8, %rsi
	jmp	.L75
.L111:
	movq	%rax, %rdx
	jmp	.L17
.L198:
.LEHB7:
	call	_ZSt28__throw_bad_array_new_lengthv@PLT
.LEHE7:
.L200:
	movq	-536(%rbp), %rax
	cmpq	$0, -280(%rbp)
	movq	(%rax), %rax
	movq	%rax, (%rdi)
	movq	-544(%rbp), %rax
	movq	%r13, 104(%rax)
	jne	.L98
	jmp	.L99
.L76:
	jne	.L74
	movq	-280(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, (%rcx)
	jmp	.L74
.L52:
	movabsq	$9223372036854775792, %rax
	movq	%rax, -256(%rbp)
	movq	%rax, %rdi
	jmp	.L53
	.p2align 4
	.p2align 3
.L199:
.LEHB8:
	call	_ZSt28__throw_bad_array_new_lengthv@PLT
.LEHE8:
.L67:
	movabsq	$9223372036854775800, %rbx
	jmp	.L68
.L192:
	movq	-296(%rbp), %rcx
	vmovss	-516(%rbp), %xmm5
	imulq	%rax, %rcx
	addq	%rdx, %rcx
	vmovss	(%r15,%rcx,4), %xmm3
	movq	%r10, %rcx
	imulq	%rax, %rcx
	addq	%rdx, %rcx
	vmovss	(%r11,%rcx,4), %xmm2
	jmp	.L108
.L15:
	movabsq	$9223372036854775792, %rbx
	jmp	.L16
.L125:
	movq	%rax, %rbx
	jmp	.L27
.L126:
	movq	%rax, %rbx
	jmp	.L86
.L123:
	movq	%rax, %rbx
	vzeroupper
	jmp	.L104
.L124:
	movq	%rax, %rbx
	jmp	.L102
.L127:
	movq	%rax, %rbx
	vzeroupper
	jmp	.L106
.L169:
	jmp	.L170
.L171:
	jmp	.L172
	.section	.gcc_except_table,"a",@progbits
.LLSDA10559:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE10559-.LLSDACSB10559
.LLSDACSB10559:
	.uleb128 .LEHB0-.LFB10559
	.uleb128 .LEHE0-.LEHB0
	.uleb128 .L169-.LFB10559
	.uleb128 0
	.uleb128 .LEHB1-.LFB10559
	.uleb128 .LEHE1-.LEHB1
	.uleb128 .L127-.LFB10559
	.uleb128 0
	.uleb128 .LEHB2-.LFB10559
	.uleb128 .LEHE2-.LEHB2
	.uleb128 .L171-.LFB10559
	.uleb128 0
	.uleb128 .LEHB3-.LFB10559
	.uleb128 .LEHE3-.LEHB3
	.uleb128 .L123-.LFB10559
	.uleb128 0
	.uleb128 .LEHB4-.LFB10559
	.uleb128 .LEHE4-.LEHB4
	.uleb128 .L124-.LFB10559
	.uleb128 0
	.uleb128 .LEHB5-.LFB10559
	.uleb128 .LEHE5-.LEHB5
	.uleb128 .L126-.LFB10559
	.uleb128 0
	.uleb128 .LEHB6-.LFB10559
	.uleb128 .LEHE6-.LEHB6
	.uleb128 .L125-.LFB10559
	.uleb128 0
	.uleb128 .LEHB7-.LFB10559
	.uleb128 .LEHE7-.LEHB7
	.uleb128 .L123-.LFB10559
	.uleb128 0
	.uleb128 .LEHB8-.LFB10559
	.uleb128 .LEHE8-.LEHB8
	.uleb128 .L124-.LFB10559
	.uleb128 0
.LLSDACSE10559:
	.text
	.cfi_endproc
	.section	.text.unlikely
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDAC10559
	.type	_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff.cold, @function
_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff.cold:
.LFSB10559:
.L174:
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	.cfi_escape 0x10,0x6,0x2,0x76,0
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	leaq	.LC1(%rip), %rdi
	vzeroupper
.LEHB9:
	call	_ZSt20__throw_length_errorPKc@PLT
.LEHE9:
.L27:
	xorl	%edx, %edx
	xorl	%esi, %esi
	xorl	%ecx, %ecx
	leaq	-160(%rbp), %rdi
	movq	%rsi, -128(%rbp)
	movq	%rdx, -160(%rbp)
	movq	%rdx, -144(%rbp)
	movl	%ecx, -136(%rbp)
	vzeroupper
	call	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
	movl	$8, %esi
.L28:
	movq	-536(%rbp), %rdi
	call	_ZdlPvm@PLT
.L106:
	movq	-496(%rbp), %rsi
	movq	-360(%rbp), %rax
	subq	%rax, %rsi
	testq	%rax, %rax
	je	.L107
	movq	%rax, %rdi
	call	_ZdlPvm@PLT
.L107:
	movq	%rbx, %rdi
.LEHB10:
	call	_Unwind_Resume@PLT
.LEHE10:
.L128:
.L172:
	movq	%rax, %rbx
	vzeroupper
.L87:
	movq	-280(%rbp), %rax
	leaq	-160(%rbp), %rdi
	movq	%rax, -160(%rbp)
	movq	-560(%rbp), %rax
	movq	%rax, -144(%rbp)
	movl	-520(%rbp), %eax
	movl	%eax, -136(%rbp)
	movq	-552(%rbp), %rax
	movq	%rax, -128(%rbp)
	call	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
	movq	-528(%rbp), %rsi
	movq	-536(%rbp), %rax
	subq	%rax, %rsi
	testq	%rax, %rax
	jne	.L28
	jmp	.L106
.L86:
	movq	-544(%rbp), %rdi
	addq	$32, %rdi
	vzeroupper
	call	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
	jmp	.L87
.L102:
	movq	-544(%rbp), %rax
	movq	72(%rax), %rdi
	movq	88(%rax), %rsi
	subq	%rdi, %rsi
	testq	%rdi, %rdi
	je	.L180
	vzeroupper
	call	_ZdlPvm@PLT
.L104:
	movq	-544(%rbp), %rdi
	addq	$32, %rdi
	call	_ZNSt13_Bvector_baseISaIbEE13_M_deallocateEv
	jmp	.L87
.L173:
	leaq	.LC1(%rip), %rdi
.LEHB11:
	call	_ZSt20__throw_length_errorPKc@PLT
.LEHE11:
.L122:
.L170:
	movq	%rax, %rbx
	vzeroupper
	jmp	.L106
.L180:
	vzeroupper
	jmp	.L104
.L175:
	leaq	.LC1(%rip), %rdi
	vzeroupper
.LEHB12:
	call	_ZSt20__throw_length_errorPKc@PLT
.LEHE12:
	.cfi_endproc
.LFE10559:
	.section	.gcc_except_table
.LLSDAC10559:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSEC10559-.LLSDACSBC10559
.LLSDACSBC10559:
	.uleb128 .LEHB9-.LCOLDB13
	.uleb128 .LEHE9-.LEHB9
	.uleb128 .L128-.LCOLDB13
	.uleb128 0
	.uleb128 .LEHB10-.LCOLDB13
	.uleb128 .LEHE10-.LEHB10
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB11-.LCOLDB13
	.uleb128 .LEHE11-.LEHB11
	.uleb128 .L122-.LCOLDB13
	.uleb128 0
	.uleb128 .LEHB12-.LCOLDB13
	.uleb128 .LEHE12-.LEHB12
	.uleb128 .L128-.LCOLDB13
	.uleb128 0
.LLSDACSEC10559:
	.section	.text.unlikely
	.text
	.size	_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff, .-_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff
	.section	.text.unlikely
	.size	_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff.cold, .-_Z13simulate_fireRK9LandscapeRKSt6vectorISt4pairImmESaIS4_EE16SimulationParamsffff.cold
.LCOLDE13:
	.text
.LHOTE13:
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.type	_GLOBAL__sub_I_rng, @function
_GLOBAL__sub_I_rng:
.LFB11643:
	.cfi_startproc
	vmovdqa	.LC14(%rip), %ymm0
	vmovdqa	%ymm0, rng(%rip)
	vzeroupper
	ret
	.cfi_endproc
.LFE11643:
	.size	_GLOBAL__sub_I_rng, .-_GLOBAL__sub_I_rng
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I_rng
	.globl	rng
	.bss
	.align 32
	.type	rng, @object
	.size	rng, 32
rng:
	.zero	32
	.section	.rodata
	.align 32
	.type	_ZL5MOVES, @object
	.size	_ZL5MOVES, 64
_ZL5MOVES:
	.long	-1
	.long	-1
	.long	-1
	.long	0
	.long	-1
	.long	1
	.long	0
	.long	-1
	.long	0
	.long	1
	.long	1
	.long	-1
	.long	1
	.long	0
	.long	1
	.long	1
	.align 32
	.type	_ZL6ANGLES, @object
	.size	_ZL6ANGLES, 32
_ZL6ANGLES:
	.long	1075235812
	.long	1078530011
	.long	1081824209
	.long	1070141403
	.long	1083624420
	.long	1061752795
	.long	0
	.long	1085271519
	.set	.LC2,.LC4
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC3:
	.long	-2147483648
	.long	0
	.long	0
	.long	0
	.align 16
.LC4:
	.long	1065353216
	.long	0
	.long	0
	.long	0
	.align 16
.LC5:
	.long	2147483647
	.long	0
	.long	0
	.long	0
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC6:
	.long	1602224127
	.align 4
.LC9:
	.long	1199570944
	.align 4
.LC11:
	.long	796917760
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32
.LC14:
	.long	12345
	.long	67890
	.long	13579
	.long	24680
	.long	11223
	.long	44556
	.long	77889
	.long	99000
	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.rel.local.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
	.align 8
	.type	DW.ref.__gxx_personality_v0, @object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
	.globl	__gxx_personality_v0
	.ident	"GCC: (Debian 14.2.0-16) 14.2.0"
	.section	.note.GNU-stack,"",@progbits
